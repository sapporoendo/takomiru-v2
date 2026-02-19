# アナログタコグラフ「走行/停止」読取り 指示書（機械実装用 v2）

## 0. この指示書の使い方（最重要）
この文書は「正（仕様の唯一の正本）」である。
実装・修正・デバッグは必ずこの文書のどこかの条項に紐づける。
出力が間違ったら、「結果を変えるためにコードをいじる」のではなく
まず「どの仕様が現実と合っていないか」を特定
仕様（パラメータ or ルール）を改定（バージョンを上げる）
DoD/回帰テストに追加
実装を追随
という順にする。

## 1. 目的（Scope）

### 1-1. 目的
円盤式アナログタコグラフ（スキャンPDF/画像）から以下を生成する。
- speed_ts：一定間隔の推定速度系列
- segments：走行/停止の区間列（現場ログに寄せる）
- break_candidate：停止区間のうち「休憩の可能性がある」候補（10分以上）

### 1-2. 禁止事項（よく事故るので明記）
速度0＝休憩（rest）と断定しない。
休憩は労務区分であり、速度だけでは確定不能。
本仕様では「停止」＋「休憩候補フラグ」まで。

## 2. 用語定義（ラベルの意味を固定）
- drive（走行）：停止条件に該当しない区間
- stop（停止イベント）：速度が0近傍の状態が 5分以上連続する区間
  - 目的：現場ログ（5分停止が多い）に寄せる／信号待ちを消す
- break_candidate（休憩候補）：stopのうち 10分以上の停止
  - 目的：「10分以上の休憩が取れているか」を見つける運用に対応
  - ただし荷待ち等も含まれるため 候補扱い
- break_likely（任意）：stopのうち 30分以上（昼休憩っぽい可能性が上がる）

## 3. 入力仕様
入力：PDF / PNG / JPG
推奨：円盤直径 1000px以上（最低600px）
許容：斜め撮影・歪み・薄い印字・汚れ
→ ただし qc_flags と confidence を必ず返す

## 4. 出力仕様（固定）

### 4-1. speed_ts（必須）
- timestamp（当日内時刻）
- speed_kmh
- quality（0〜1）
- source（例：trace_extracted / interpolated / missing）

### 4-2. segments（必須）
各セグメントに以下を含める：
- start_time, end_time, duration_sec
- label: drive または stop
- break_candidate: boolean（stopのみ意味を持つ）
- break_likely: boolean（任意）
- confidence（0〜1）
- reason（例：stop_by_threshold_5min / micro_drive_absorbed など）

### 4-3. debug（推奨）
- params_used（中心座標、速度帯、しきい値）
- qc_flags（後述）
- overlay_image（中心・速度点・停止区間を描画）

## 5. 固定パラメータ（初期値：現場ログ寄せ）
変更が必要になったら「変更履歴（セクション11）」に必ず書く。

sampling_interval_sec: 10
chart_hours: 24

# hysteresis for stop/drive
v_stop_in: 2        # stop入り（0近傍）
v_stop_out: 5       # stop抜け（渋滞チラつき防止）

# segment thresholds
t_min_stop_sec: 300     # stop採用は5分以上
t_min_drive_sec: 60     # 1分未満のdriveはノイズとして吸収

# break flags
t_break_candidate_sec: 600  # 10分以上を休憩候補
t_break_likely_sec: 1800    # 30分以上を休憩らしい（任意）

# smoothing
median_window: 5         # 50秒相当

# speed scale (暫定)
speed_vmax_kmh: 100

# time anchor
time_anchor_mode: manual_or_template

## 6. アルゴリズム（人間の読み方を機械化：処理順が命）

### 6-1. PDF→画像化
PDF各ページを 300dpi相当以上で画像化

### 6-2. ROI抽出（円盤領域）
グレースケール化＋コントラスト補正
二値化→最大の円形近傍領域を円盤候補として抽出
円盤外周＋余白5〜10%でROI切り出し
失敗時：qc_flags.no_disc_detected

### 6-3. 中心点推定（最重要）
優先順位：
円盤外周（outer circle）を検出→中心（outer-first）
補助：中央穴（spindle、暗い円）検出→中心（任意）
だめなら同心円検出→同心性最大の中心
不確実：中心候補差が大きい → qc_flags.center_uncertain

### 6-4. 速度帯（speed band）の確定
速度帯（0–120km/hリング）は、円盤外周半径 outer_radius を基準に固定の円環ROIとして切り出す。

- R = outer_radius
- r_in = 0.55 * R
- r_out = 0.86 * R

ROI外は強制的にマスク（ピクセル=0）し、以降の針抽出はROI内限定で行う。
outer_radius が不確実/未推定の場合：qc_flags.speed_band_uncertain
不確実：qc_flags.speed_band_uncertain

### 6-5. 角度サンプリング
sampling_interval_sec に従い1周を等分し角度列 θ_i を作る

### 6-6. 速度線抽出（角度→半径→速度）
各角度 θ_i で：
放射線上 [r_in, r_out] を走査
インクらしさスコア S(r) 最大を線位置 r* とする
連続性制約：r* が前点から急ジャンプなら次点候補を優先
r* を speed_raw に換算（暫定：線形マップ）
移動中央値 median_window で speed_smooth
各点に quality を付与（スコア/安定性で算出）
品質劣化：qc_flags.trace_faint / qc_flags.trace_noisy

### 6-7. drive/stop判定（ヒステリシス）
ステートマシン：
drive → speed_smooth <= v_stop_in でstopへ
stop → speed_smooth >= v_stop_out でdriveへ

### 6-8. セグメント生成（最小長ルールが核心）
ラベル列を連結して区間化
stop採用条件：stop.duration >= t_min_stop_sec(=300) のみ stopとして出力
5分未満のstopは 出力しない（前後driveへ吸収）
micro-drive除去：drive.duration < t_min_drive_sec(=60) は前後stopへ吸収
これで線のブレ由来の細切れdriveを消す
ただし 5分driveは残す（現場ログに普通に出るため）

### 6-9. 休憩候補フラグ付与（stopの属性）
break_candidate = (stop.duration >= 600 sec)
break_likely = (stop.duration >= 1800 sec)（任意）

### 6-10. 時刻合わせ（MVPは手動アンカー優先）
time_anchor_mode=manual_or_template
MVP：ユーザーが「この角度がXX:XX」を1点指定し、θ0 を決定
自動（テンプレ/OCR）は後で追加可
失敗時は時刻不確実として qc_flags.time_anchor_uncertain

## 7. QCフラグ（必須）
- no_disc_detected
- center_uncertain
- speed_band_uncertain
- trace_faint
- trace_noisy
- scale_or_center_error（Vmax超過多発など）
- chattering（stop/driveが短周期交互）
- time_anchor_uncertain

## 8. Confidence（0〜1）算出（固定ルール）
初期値1.0から減点：
- center_uncertain -0.2
- speed_band_uncertain -0.2
- trace_faint -0.2
- trace_noisy -0.2
- chattering -0.2
- time_anchor_uncertain -0.2
下限0

## 9. DoD（受け入れ基準：これ満たさないと未完）
- stopセグメントは 5分未満が出力されない
- break_candidateは 10分以上のstopでtrueになる
- micro-drive（<60秒）は消えるが、5分driveは残る
- overlayで中心・速度点・stop区間が確認できる
- 失敗時にQCフラグが立つ（沈黙しない）

## 10. 回帰テスト（ログで検証する“期待特性”）
あなたが貼った日次ログを「期待値」として使う。ここを増やすほど精度が安定する。
各日の期待特性（例）：
- stopの多くが 5分/10分/15分単位で出ている（過剰に細切れにならない）
- 10分以上stopが複数回ある日（例：R6.3.11, R6.3.13）で break_candidate=true が適切に付く
- 長時間stop（例：R6.3.1の9時間半）で break_likely=true（採用するなら）
※具体の“本数・合計時間”を固定したい場合は、ここに数値期待を追記する。

## 11. 変更履歴（ここが仕様運用の肝）
- v1.0：stop>=5分で現場ログ寄せ（信号待ち除外）
- v1.1：stopに break_candidate(>=10分) を追加。drive側は5分を残すため t_min_drive_sec=60 に固定。

以後、変更はこの形式で追記：
vX.Y：何を、なぜ変えたか（例：閾値を300→600に変更、理由：監査基準に合わせる）

## 12. 修正手順（出力が間違っていたらこう直す）
症状を分類
- stopが細切れすぎる → t_min_stop を上げる or 平滑化強める or v_stop_in を下げる
- stopが消えすぎる → t_min_stop を下げる or v_stop_in を上げる
- driveが細切れ → t_min_drive を上げる（ただし5分driveを壊さない上限は120秒程度）
- break_candidateが多すぎる → t_break_candidate を15分等に上げる
仕様（この文書）を先に改定してバージョンを上げる
回帰テストに“その症状の例”を1つ追加
実装を変更

---

以後、アナログタコグラフ解析は本ファイル（Tachograph_Spec.md）を唯一の仕様書として扱い、実装・修正・出力の判断は必ず当仕様に従うこと。出力が期待と違う場合は、まず仕様のどの項目が原因かを特定し、仕様の改定→回帰テスト追加→実装変更の順で行うこと。
