# 预测
python main.py --task='prediction'  --base_url='https://your_base_url'  --model_name='your_model_name' --api_key='your_api_key'

# SQL执行结果转化为TimeSeries
python main.py --task='transform' --model_name='your_model_name'


# 计算指标
python main.py --task='eval' --model_name='your_model_name'