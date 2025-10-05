use base::detect_and_click::detect_and_click;
use eframe::{NativeOptions, egui};
use egui::SliderClamping;

pub struct AutoAIApp {
    running: bool,
    model_path: String,
    image_path: String,
    status: String,
    conf_thres: f32,
}

impl Default for AutoAIApp {
    fn default() -> Self {
        Self {
            running: false,
            model_path: "best.onnx".into(), // 根目錄 best.onnx
            image_path: "/Users/terry/Desktop/ScreenShot/Screenshot  2025-09-28 at 21.23.55.png"
                .into(), // 放你的螢幕截圖路徑
            status: "Idle".into(),
            conf_thres: 0.9, // YOLO 置信度門檻
        }
    }
}

impl eframe::App for AutoAIApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("RVF Work Flow Designer");
            ui.separator();
            ui.label("Model Path (.onnx):");
            ui.text_edit_singleline(&mut self.model_path);
            ui.label("Image Path (whole screenshot .png/.jpg ) :");
            ui.text_edit_singleline(&mut self.image_path);
            ui.horizontal(|ui| {
                ui.label(format!("Conf Thres: {:.2}", self.conf_thres));
                ui.add(
                    egui::Slider::new(&mut self.conf_thres, 0.05..=0.95)
                        .clamping(SliderClamping::Always),
                );
            });
            ui.separator();
            if ui.button("Execute YOLO recognize and click").clicked() {
                self.running = true;
                // 這裡呼叫 AI+鼠標 事件，例如 spawn detect_and_click()
                self.status = "Processing...".into();
                let model = self.model_path.trim().to_string();
                let img = self.image_path.trim().to_string();
                let thres = self.conf_thres;
                // Demo：同步呼叫；若要避免 UI 阻塞，之後可改 thread + channel
                match detect_and_click(&model, &img, thres) {
                    Ok(msg) => self.status = msg,
                    Err(e) => self.status = format!("Error: {e}"),
                }
                self.running = false;
            }

            if self.running {
                ui.label("Processing...");
            } else {
                ui.label(format!("Status: {}", self.status));
            }

            ui.separator();
            ui.label("Show flow here");
        });
    }
}

fn main() {
    let options = NativeOptions::default();
    let _ = eframe::run_native(
        "Rust Vision Flow",
        options,
        Box::new(|_cc| Ok(Box::new(AutoAIApp::default()))),
    );
}