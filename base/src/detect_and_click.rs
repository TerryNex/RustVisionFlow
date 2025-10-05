use anyhow::{Result, anyhow};
use image::{GenericImageView, imageops::FilterType};
use onnxruntime::{
    GraphOptimizationLevel, LoggingLevel,
    environment::Environment,
    ndarray::{Array2, Array3, Array4, Axis, Ix3, IxDyn, s},
    tensor::OrtOwnedTensor,
};
use rdev::{Button, EventType, SimulateError, simulate};
use std::{path::Path, sync::Arc, time::Duration};

const INPUT_SIZE: u32 = 640;

// 簡化版 NMS
fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    // xywh -> xyxy
    let (ax1, ay1, aw, ah) = (a[0] - a[2] * 0.5, a[1] - a[3] * 0.5, a[2], a[3]);
    let (ax2, ay2) = (ax1 + aw, ay1 + ah);
    let (bx1, by1, bw, bh) = (b[0] - b[2] * 0.5, b[1] - b[3] * 0.5, b[2], b[3]);
    let (bx2, by2) = (bx1 + bw, by1 + bh);

    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn nms(mut boxes: Vec<([f32; 4], f32, usize)>, iou_thres: f32) -> Vec<([f32; 4], f32, usize)> {
    // 依 score 由高到低
    boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut keep: Vec<([f32; 4], f32, usize)> = Vec::new();
    'outer: for i in 0..boxes.len() {
        for k in &keep {
            if iou(boxes[i].0, k.0) > iou_thres {
                continue 'outer;
            }
        }
        keep.push(boxes[i]);
    }
    keep
}

/// 回傳字串會包含被點擊的類別與座標
pub fn detect_and_click(model_path: &str, image_path: &str, conf_thres: f32) -> Result<String> {
    // 1) 載入圖片並記錄縮放比例
    let img_path = Path::new(image_path);
    if !img_path.exists() {
        return Err(anyhow!("Image not found: {}", image_path));
    }
    let img = image::open(img_path)?;
    let (orig_w, orig_h) = img.dimensions();

    // 2) 轉成 640x640、RGB、NCHW、[0,1]
    let resized = img
        .resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle)
        .to_rgb8();
    let mut chw = Array3::<f32>::zeros((3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for (y, row) in resized.rows().into_iter().enumerate() {
        for (x, px) in row.into_iter().enumerate() {
            let [r, g, b] = px.0;
            chw[[0, y, x]] = r as f32 / 255.0;
            chw[[1, y, x]] = g as f32 / 255.0;
            chw[[2, y, x]] = b as f32 / 255.0;
        }
    }
    let input: Array4<f32> = chw.insert_axis(Axis(0)); // [1,3,640,640]

    // 3) 建立 ORT 環境與 session
    let env = Arc::new(
        Environment::builder()
            .with_name("rvf")
            .with_log_level(LoggingLevel::Warning)
            .build()?,
    );
    let mut session = env
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file(model_path)?;

    // 4) 推論
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![input])?;
    if outputs.is_empty() {
        return Err(anyhow!("No outputs from ONNX session"));
    }

    // 5) 解析 YOLOv8 輸出（動態通道數）
    // 可能形狀： [1, C, N] 或 [1, N, C]，其中 C = 4 + num_classes，單類別時 C=5
    let out = outputs[0].view();
    let shape = out.shape();
    if shape.len() != 3 {
        return Err(anyhow!("Unsupported output shape: {:?}", shape));
    }

    // 決定 C 與 N 的軸向
    let (c_axis, n_axis, c, n) = if shape[1] >= 5 {
        (1, 2, shape[1], shape[2]) // [1, C, N]
    } else if shape[2] >= 5 {
        (2, 1, shape[2], shape[1]) // [1, N, C]
    } else {
        return Err(anyhow!("Expected C>=5 (got shape {:?})", shape));
    };

    // 讀取工具：依軸向取值
    let get = |i_n: usize, i_c: usize| -> f32 {
        if c_axis == 1 {
            out[[0, i_c, i_n]]
        } else {
            out[[0, i_n, i_c]]
        }
    };

    // 6) 取 conf 與 class（支援單類別/多類別）
    let mut candidates: Vec<([f32; 4], f32, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let cx = get(i, 0);
        let cy = get(i, 1);
        let w = get(i, 2);
        let h = get(i, 3);

        let cls_channels = c - 4;
        let (best_cls, best_score) = if cls_channels == 1 {
            // 單類別：第 5 維就是分數
            (0usize, get(i, 4))
        } else {
            // 多類別：在 [4 .. 4+num_classes) 中取最大
            let mut best_idx = 0usize;
            let mut best = f32::MIN;
            for k in 0..cls_channels {
                let s = get(i, 4 + k);
                if s > best {
                    best = s;
                    best_idx = k;
                }
            }
            (best_idx, best)
        };

        if best_score >= conf_thres {
            candidates.push(([cx, cy, w, h], best_score, best_cls));
        }
    }

    if candidates.is_empty() {
        return Ok(format!("No detection >= {:.2}", conf_thres));
    }

    // 7) NMS（簡化）
    let kept = nms(candidates, 0.45);
    let (best_box, best_conf, best_cls) = kept[0];

    // 8) 換算到原圖座標（YOLOv8 輸出是以模型輸入 640 為座標）
    let scale_x = orig_w as f32 / INPUT_SIZE as f32;
    let scale_y = orig_h as f32 / INPUT_SIZE as f32;
    let click_x = (best_box[0] * scale_x) as f64;
    let click_y = (best_box[1] * scale_y) as f64;

    // 9) 送滑鼠事件（rdev 需要 &EventType）
    let send = |e: EventType| -> Result<()> {
        simulate(&e).map_err(|e| anyhow!("simulate error: {:?}", e))
    };
    // 移動 → 按下 → 放開
    send(EventType::MouseMove {
        x: click_x,
        y: click_y,
    })?;
    std::thread::sleep(Duration::from_millis(10));
    send(EventType::ButtonPress(Button::Left))?;
    std::thread::sleep(Duration::from_millis(10));
    send(EventType::ButtonRelease(Button::Left))?;

    Ok(format!(
        "Clicked class={} conf={:.2} at ({:.0},{:.0})",
        best_cls, best_conf, click_x, click_y
    ))
}

#[allow(dead_code)]
/// 以 YOLOv8 ONNX 執行偵測，點選最高分框中心。
/// - `model_path`：.onnx 路徑（建議 ./best.onnx）
/// - `image_path`：整張螢幕截圖（點擊會以原圖座標為準）
/// - `conf_thres`：分數門檻（0.05~0.9）
///
/// 回傳簡短狀態字串，便於 GUI 顯示。
pub fn detect_and_click_v1(
    model_path: &str,
    image_path: &str,
    conf_thres: f32,
) -> Result<String, Box<dyn std::error::Error>> {
    if image_path.trim().is_empty() {
        return Err("Image path is empty".into());
    }
    let img_orig = image::open(image_path)?;
    let (ow, oh) = img_orig.dimensions();
    // ===== 前處理：resize 到 640x640、RGB、CHW、[0,1] =====
    let img = img_orig
        .resize_exact(640, 640, FilterType::Triangle)
        .to_rgb8();
    let mut chw = Array4::<f32>::zeros((1, 3, 640, 640));
    // HWC(u8) -> CHW(f32)
    for y in 0..640 {
        for x in 0..640 {
            let p = img.get_pixel(x, y).0;
            chw[[0, 0, y as usize, x as usize]] = p[0] as f32 / 255.0; // R
            chw[[0, 1, y as usize, x as usize]] = p[1] as f32 / 255.0; // G
            chw[[0, 2, y as usize, x as usize]] = p[2] as f32 / 255.0; // B
        }
    }
    // 若模型檔不存在，直接回退「點圖中心」
    let has_onnx = !model_path.trim().is_empty() && Path::new(model_path).is_file();
    let (mut click_x, mut click_y) = ((ow as f64) / 2.0, (oh as f64) / 2.0);
    if has_onnx {
        // ===== 載入 ONNX =====
        let env = Environment::builder().with_name("rvf-yolov8").build()?;
        let mut session = env
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_model_from_file(model_path)?;
        // ===== 推論 =====
        // onnxruntime 的 run() 可直接吃 ndarray，回傳 OrtOwnedTensor（ndarray 檢視）
        let outputs: Vec<OrtOwnedTensor<f32, IxDyn>> = session.run(vec![chw])?;
        let out = &outputs[0]; // 形狀約為 [1, 84, N]（N=8400）
        // ===== 解析輸出：找分數最高的框 =====
        // 約定：前4為(cx, cy, w, h)，之後為各類別分數；取 max class score 當置信度
        let shape = out.view().shape().to_vec(); // [1, 84, N]
        if shape.len() == 3 && shape[1] >= 5 {
            let n = shape[2];
            let mut best_score = -1.0_f32;
            let mut best_cx = 0f32;
            let mut best_cy = 0f32;
            let mut best_w = 0f32;
            let mut best_h = 0f32;
            for i in 0..n {
                // 取各類別的最大分數
                let mut s = 0f32;
                for cls in 4..shape[1] {
                    let v = out[[0, cls, i]];
                    if v > s {
                        s = v;
                    }
                }
                if s >= conf_thres && s > best_score {
                    best_score = s;
                    best_cx = out[[0, 0, i]];
                    best_cy = out[[0, 1, i]];
                    best_w = out[[0, 2, i]];
                    best_h = out[[0, 3, i]];
                }
            }
            if best_score >= 0.0 {
                // (cx,cy,w,h) 是以 640x640 為座標，轉回原圖座標
                let sx = (ow as f32) / 640.0;
                let sy = (oh as f32) / 640.0;
                let cx = best_cx * sx;
                let cy = best_cy * sy;
                click_x = cx as f64;
                click_y = cy as f64;
            }
        }
    }
    // ===== 發送滑鼠事件 =====
    // 注意：macOS 需在「系統設定→隱私權與安全性→輔助使用」允許；Wayland 桌面對全域注入有限制（X11 較寬鬆）。
    simulate(&EventType::MouseMove {
        x: click_x,
        y: click_y,
    })
    .map_err(|_e| SimulateError)?;
    simulate(&EventType::ButtonPress(Button::Left)).map_err(|_e| SimulateError)?;
    simulate(&EventType::ButtonRelease(Button::Left)).map_err(|_e| SimulateError)?;
    let msg = format!("Clicked at ({:.0}, {:.0})", click_x, click_y);
    Ok(msg.into())
}