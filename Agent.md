# AGENT 開發規範（For AI Code Agent / Codex）

代碼文件的注釋和內容使用英文

## 核心設計指引

- **模組化：** 每一功能元件（任務步驟、AI 推論、滑鼠控制等）均需抽象出 trait，方便擴展/mock/單測。
- **流程消息流：** 任務步驟均以 message/event 形式傳遞，允許插拔、序列化、condition 判斷。
- **egui 介面契約：** 所有流程配置皆須支援編輯 GUI 操作與即時反饋。
- **資料/腳本模型：** 所有 step/workflow 支援 serde-json（或 ron）序列化和版本升級相容。
- **文檔完善：** 自動生成的代碼需加註必要文檔與範例。

## 預期參與方式

- 新增/優化自動化 action（如複雜識別、影像強化、對話介面等）。
- 新的流程步驟請實作 `WorkflowStep` trait，並在 workflow module 註冊。
- AI agent 可協助撰寫流程解析、腳本轉換、UI template 自動化配置。
- 支援自定義腳本語法（可 JSON/YAML/DSL）自動轉成消息流節點。
- 每組件須留單元測試接口，便於 CI 自動測試/維護。

## 範例 Trait

```
pub trait WorkflowStep: Serialize + DeserializeOwned {
fn name(&self) -> &str;
fn execute(&mut self, context: &mut WorkflowContext) -> StepResult;
// Support condition/loop/side effect for flow control
}
```