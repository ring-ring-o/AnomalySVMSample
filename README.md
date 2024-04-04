# AnomalySVMSample
アノマリー検知の動作確認

## 実行方法
1. devcontainerを開く
2. `pnpm install`を実行する
3. `pnpm run python create_sample.py`を実行しサンプルデータを作成する
4. `pnpm run python learn_svm.py`を実行しsvmを学習する
5. 完了すると/workspace/model.pickleが作成される
