# Flexible Framework for LSTM Stock Prediction

This project provides a flexible and powerful framework for designing, training, and evaluating LSTM models for stock price prediction. It is structured to facilitate rapid experimentation and ensure reproducibility through automated model card generation.

---

<details>
<summary>🇬🇧 English</summary>

### Features
- **Config-Driven Model Architecture**: Easily design complex LSTM architectures by simply editing a configuration file. No code changes needed.
- **Automated Experiment Tracking**: Each training run automatically generates a "Model Card" in its own timestamped folder.
- **Comprehensive Model Cards**: Each card includes the model's architecture summary, performance metrics, training history plot, and the exact configuration used.
- **Modular & Clean Code**: The project is organized into logical modules for data loading, model building, training, and prediction.

### Project Structure
```
├── data/              # Place your stock .csv files here
├── models/            # Trained models and their cards are saved here (ignored by .gitignore)
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── src/
│   ├── config.py      # Main configuration file for your experiments
│   ├── data_loader.py # Handles data loading and preprocessing
│   └── model.py       # The class-based LSTMBuilder
├── train.py           # Script to run model training
├── predict.py         # Script to make predictions with a trained model
├── requirements.txt   # Project dependencies
└── README.md
```

### How to Use

#### 1. Setup
First, clone the repository and install the required packages.
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

#### 2. Configure Your Experiment
Open `src/config.py` to set up your experiment. The most important part is `MODEL_ARCHITECTURE`.

**Example: Designing a new model**
To change the model, simply edit the `layers` list. For instance, to create a simpler model:
```python
# src/config.py

MODEL_ARCHITECTURE = {
    'name': 'Simple_LSTM_v2', # Give it a new name
    'layers': [
        {'type': 'lstm', 'units': 50, 'return_sequences': False}, # Only one LSTM layer
        {'type': 'dense', 'units': 25, 'activation': 'relu'},
        {'type': 'dense', 'units': 1} # Output layer
    ]
}
```
You can also change the `TICKER`, `WINDOW_SIZE`, `TRAINING_PARAMS`, etc., in the same file.

#### 3. Train the Model
Run the training script from your terminal:
```bash
python train.py
```
The script will create a new folder in `models/` named something like `Simple_LSTM_v2_20231027-103000`. Inside, you will find:
- `model.h5`: The trained model file.
- `model_card.md`: A detailed report of your experiment.
- `training_loss.png`: A plot of the training/validation loss.

#### 4. Make a Prediction
To use the latest trained model for prediction:
```bash
python predict.py
```
To use a specific model, pass the path to its folder:
```bash
python predict.py --model_folder models/Simple_LSTM_v2_20231027-103000
```
A plot named `prediction_vs_actual.png` will be saved in that model's folder.

### Recommended Hardware Specifications
- **Minimum**:
    - CPU: 4-core processor
    - RAM: 8 GB
    - GPU: Not required, but training will be slow.
- **Recommended for Active Development**:
    - CPU: 6-core processor or better
    - RAM: 16 GB or more
    - GPU: NVIDIA GeForce RTX 3060 (6GB VRAM) or better. A GPU is highly recommended to significantly speed up training times.

</details>

<details>
<summary>🇨🇳 简体中文</summary>

### 功能亮点
- **配置驱动的模型架构**：只需编辑配置文件即可轻松设计复杂的 LSTM 架构，无需修改任何代码。
- **自动化实验追踪**：每次训练都会在一个带时间戳的专属文件夹中，自动生成"模型卡片"。
- **全面的模型卡片**：每张卡片都包含模型结构摘要、性能指标、训练历史图表以及该次训练所使用的完整配置。
- **模块化清晰代码**：项目被组织成用于数据加载、模型构建、训练和预测的逻辑模块。

### 项目结构
```
├── data/              # 在此处放置你的股票 .csv 文件
├── models/            # 训练好的模型及其卡片保存在此 (被 .gitignore 忽略)
├── notebooks/         # 用于探索和分析的 Jupyter notebooks
├── src/
│   ├── config.py      # 用于实验的主配置文件
│   ├── data_loader.py # 处理数据加载和预处理
│   └── model.py       # 基于类的 LSTMBuilder
├── train.py           # 运行模型训练的脚本
├── predict.py         # 使用已训练模型进行预测的脚本
├── requirements.txt   # 项目依赖
└── README.md
```

### 使用方法

#### 1. 环境设置
首先，克隆仓库并安装所需的包。
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

#### 2. 配置你的实验
打开 `src/config.py` 来设置你的实验。最重要的部分是 `MODEL_ARCHITECTURE`。

**示例：设计一个新模型**
要更改模型，只需编辑 `layers` 列表。例如，要创建一个更简单的模型：
```python
# src/config.py

MODEL_ARCHITECTURE = {
    'name': 'Simple_LSTM_v2', # 给它一个新名字
    'layers': [
        {'type': 'lstm', 'units': 50, 'return_sequences': False}, # 只有一个 LSTM 层
        {'type': 'dense', 'units': 25, 'activation': 'relu'},
        {'type': 'dense', 'units': 1} # 输出层
    ]
}
```
你也可以在同个文件中更改 `TICKER`、`WINDOW_SIZE`、`TRAINING_PARAMS` 等。

#### 3. 训练模型
在你的终端中运行训练脚本：
```bash
python train.py
```
该脚本将在 `models/` 中创建一个新文件夹，名称类似于 `Simple_LSTM_v2_20231027-103000`。在里面，你会找到：
- `model.h5`: 训练好的模型文件。
- `model_card.md`: 关于你实验的详细报告。
- `training_loss.png`: 训练/验证损失的图表。

#### 4. 进行预测
要使用最新训练的模型进行预测：
```bash
python predict.py
```
要使用特定的模型，请传递其文件夹的路径：
```bash
python predict.py --model_folder models/Simple_LSTM_v2_20231027-103000
```
一个名为 `prediction_vs_actual.png` 的图表将被保存在该模型的文件夹中。

### 推荐硬件规格
- **最低配置**：
    - CPU: 4核处理器
    - 内存: 8 GB
    - GPU: 非必需，但训练会很慢。
- **推荐用于积极开发**：
    - CPU: 6核或更好的处理器
    - 内存: 16 GB 或更多
    - GPU: NVIDIA GeForce RTX 3060 (6GB VRAM) 或更好。强烈推荐使用 GPU 以显著加快训练速度。

</details>


<details>
<summary>🇹🇼 繁體中文</summary>

### 功能亮點
- **設定驅動的模型架構**：只需編輯設定檔即可輕鬆設計複雜的 LSTM 架構，無需修改任何程式碼。
- **自動化實驗追蹤**：每次訓練都會在一個帶時間戳的專屬資料夾中，自動生成「模型卡片」。
- **全面的模型卡片**：每張卡片都包含模型結構摘要、效能指標、訓練歷史圖表以及該次訓練所使用的完整設定。
- **模組化清晰程式碼**：專案被組織成用於資料載入、模型建構、訓練和預測的邏輯模組。

### 專案結構
```
├── data/              # 在此處放置你的股票 .csv 檔案
├── models/            # 訓練好的模型及其卡片保存在此 (被 .gitignore 忽略)
├── notebooks/         # 用於探索和分析的 Jupyter notebooks
├── src/
│   ├── config.py      # 用於實驗的主設定檔
│   ├── data_loader.py # 處理資料載入和預處理
│   └── model.py       # 基於類別的 LSTMBuilder
├── train.py           # 執行模型訓練的腳本
├── predict.py         # 使用已訓練模型進行預測的腳本
├── requirements.txt   # 專案依賴
└── README.md
```

### 使用方法

#### 1. 環境設定
首先，克隆儲存庫並安裝所需的套件。
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

#### 2. 設定你的實驗
開啟 `src/config.py` 來設定你的實驗。最重要的部分是 `MODEL_ARCHITECTURE`。

**範例：設計一個新模型**
要更改模型，只需編輯 `layers` 列表。例如，要建立一個更簡單的模型：
```python
# src/config.py

MODEL_ARCHITECTURE = {
    'name': 'Simple_LSTM_v2', # 給它一個新名字
    'layers': [
        {'type': 'lstm', 'units': 50, 'return_sequences': False}, # 只有一個 LSTM 層
        {'type': 'dense', 'units': 25, 'activation': 'relu'},
        {'type': 'dense', 'units': 1} # 輸出層
    ]
}
```
你也可以在同個檔案中更改 `TICKER`、`WINDOW_SIZE`、`TRAINING_PARAMS` 等。

#### 3. 訓練模型
在你的終端機中執行訓練腳本：
```bash
python train.py
```
該腳本將在 `models/` 中建立一個新資料夾，名稱類似於 `Simple_LSTM_v2_20231027-103000`。在裡面，你會找到：
- `model.h5`: 訓練好的模型檔案。
- `model_card.md`: 關於你實驗的詳細報告。
- `training_loss.png`: 訓練/驗證損失的圖表。

#### 4. 進行預測
要使用最新訓練的模型進行預測：
```bash
python predict.py
```
要使用特定的模型，請傳遞其資料夾的路徑：
```bash
python predict.py --model_folder models/Simple_LSTM_v2_20231027-103000
```
一個名為 `prediction_vs_actual.png` 的圖表將被儲存在該模型的資料夾中。

### 推薦硬體規格
- **最低配置**：
    - CPU: 4核心處理器
    - 記憶體: 8 GB
    - GPU: 非必需，但訓練會很慢。
- **推薦用於積極開發**：
    - CPU: 6核心或更好的處理器
    - 記憶體: 16 GB 或更多
    - GPU: NVIDIA GeForce RTX 3060 (6GB VRAM) 或更好。強烈推薦使用 GPU 以顯著加快訓練速度。

</details> 