# TorchTitan GitCode Hosted 控制候选：尚未上线

本候选在个人 `gitcode-forward` 分支开发，不是官方 GitCode 控制发布，不代表 U=C 已核验，也不代表完整 GitCode Forward 已验收。

## 已恢复的前置与新增实现

恢复已合入 PR #11 对应 `f20d073aec6bb7543b2e7db706d1e01b5298c8ca` 的统一 Smoke/Nightly Workflow、Prepare/Waiter、受信 schema-3 目录/合同及相关测试依赖。只移植新增/修改文件，不执行该 PR 的旧入口删除，不改 Required Checks，不执行 Yellow。

`.github/scripts/lite_actions/gitcode/hosted.py` 提供 schema-4 Hosted Prepare/Waiter 边界：校验实际仓库 ID、默认分支、不可变控制 SHA/Workflow SHA/路径、Run ID/Attempt；拒绝原生 Job rerun。Prepare 委托已验证的 Worker SDK Producer；Waiter 只消费完整验证的请求、固定 Check IDs、receipt 与当前版本回调。读取或轮询越过 deadline 不能返回 PASS。

`upload_artifact.cjs` 一次只上传一个明确的普通文件，返回不可变 Artifact ID；输出不明确时交由 Producer 的持久化未知状态恢复，不盲目重复上传。Node 子进程只得到 Artifact runtime 所需环境，不继承 GitHub PAT、GitCode Token、App 私钥或 NODE_OPTIONS。

依赖 `@actions/artifact` 固定为 6.3.1，全部传递依赖由 package-lock.json 固定，使用动态 ESM import；安装命令为 `npm ci --ignore-scripts --no-audit --no-fund`。Node 24，仅在可信控制目录安装；node_modules 不提交、不随 Source/Runner Bundle 传输。

## 验证命令

Worker SDK 必须是审核后固定提交，当前配套验证版本为 `depeng1994/lite-actions@d9444a26911864fc79f1bf756ff5545014e749a6`。测试运行时通过 PYTHONPATH 指向该 SDK。不要从 Dispatch/PR 输入选择 SDK 或控制 Python 模块。

```bash
PYTHONPATH=/path/to/pinned-lite-actions python3 -m unittest discover -s tests/unit_tests/tooling/gitcode_hosted -v
python3 -m unittest discover -s .github/scripts/lite_actions/gitcode/tests -v
python3 -m pytest --noconftest -q -o addopts='' tests/unit_tests/tooling/lite_actions tests/unit_tests/test_lite_actions_dispatch_smoke.py tests/unit_tests/models/deepseek_v4/test_nightly_flash_recipes.py
```

最后一组是控制/脚本定向回归，使用 --noconftest 避免加载产品 NPU 全局 fixture；其中 runner_port_forwarding 仍需真正的 PyTorch，不能用假模块代替。以上不是 NPU 或 Hosted 实际 API 验收。

## 仍需完成的装配与发布门槛

这次恢复的 Workflow 仍是已验收的 schema-3 入口，**尚未新增 GitCode source_provider 分支**。不发布引用不存在的 CLI/平台适配器的半套 schema-4 Workflow。

Hosted 真实编排还需接入可信发布配置、固定 SDK、GitCode 认证/事件适配、同一专用 App 的短期身份，以及已核验的当前发布权回调；不得仅比较当前 H/B 就声称同 H/B 的旧规格仍有发布权。Bridge 与 Blue 的事件顺序/代际凭据必须来自真实可信平台记录，而非输入自报 revision。

上传器必须在 GitHub 的 JavaScript Action runtime 中运行，使用平台提供的 ACTIONS_RUNTIME_TOKEN/ACTIONS_RESULTS_URL。不能假定普通 shell run 步骤天然有这些环境，不能用 PAT 替代 Artifact runtime Token。尚无实际 Hosted 上传/丢响应平台记录。

2026-11-02 对此锁文件执行 npm audit：0 high、0 critical、7 moderate（archiver/glob/minimatch 依赖链）。不能宣称依赖审计全绿。当前接口仅传固定单文件路径，不能让 PR 指定 glob；仍需发布前评估或通过审核的依赖升级，不盲目应用降级建议。

官方 GitCode 上游由用户审查合入；之后按原设计核验自动或本次手动同步的 U=C。当前候选不得直接作为组织 GitHub 独有控制补丁上线。GitCode 凭证调用、专用 App 创建/更新权限、状态回写、实际 NPU 验收和生产迁移/回滚均必须另有证据。
