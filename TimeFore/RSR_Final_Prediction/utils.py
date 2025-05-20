import json
import os
import ast

# 将生成的表格名称存储到json文件当中
class PredictResultStorage:
    def __init__(self, file_name='testdata.json', save_lock=None):
        self.current_data = {}
        self.file_name = file_name
        self.save_lock = save_lock or threading.Lock()
    def set_data(self, data_dict):
        self.current_data.update(data_dict)
    def save_data(self):
        with self.save_lock:
            if os.path.exists(self.file_name):
                with open(self.file_name, 'r+') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                    data.append(self.current_data)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
            else:
                with open(self.file_name, 'w') as f:
                    json.dump([self.current_data], f, indent=4)
            self.current_data = {}



def align_tokens_with_query(tokens, query):
    query = list(query)
    new_tokens = []
    for i, token in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        elif token == query[0]:
            new_tokens.append(token)
            query = query[1:]
        elif '##' in token:
            token = token[2:]
            new_tokens.append(token)
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
                    
        elif '[UNK]' == token:
            end_index = query.index(tokens[i+1])
            unk = ''.join(query[0:end_index])
            new_tokens.append(unk)
            query = query[end_index:]

        elif len(token)>1:
            new_tokens.append(token)
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
    return new_tokens    

def restore_keywords_from_tokens(tokens, token_slot):
    keywords = []
    current_tokens = []
    current_label = None
    token_slot = token_slot[1:-1]

    for token, slot in zip(tokens, token_slot):
        if slot.startswith('B-'):
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
            current_label = slot[2:]
            current_tokens.append(token)
        elif slot.startswith('I-') and current_label == slot[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
                current_label = None

    if current_tokens:
        keywords.append((''.join(current_tokens), current_label))

    return keywords

def restore_keywords_from_query(query, slots):
    keywords = []
    current_tokens = []
    current_label = None
    query = list(query)
    if slots[0] == '[CLS]':
        slots = slots[1:-1]

    for token, slot in zip(query, slots):
        if slot.startswith('B-'):
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
            current_label = slot[2:]
            current_tokens.append(token)
        elif slot.startswith('I-') and current_label == slot[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
                current_label = None

    if current_tokens:
        keywords.append((''.join(current_tokens), current_label))

    return keywords

def convert_property_data(data_dict):

    result = []
    
    # 定义月份格式
    months = [f"2024年{i}月" for i in range(1, 13)]
    if len(data_dict.keys()) == 1:
        # 遍历字典中的每个小区
        for community, values in data_dict.items():
            # 为每个月创建一个元组
            values = values[0]
            for i, value in enumerate(values):
                if i < len(months):  # 确保索引不超出月份列表范围
                    # 只添加非零值到结果列表中
                    result.append((months[i], value))
    else:
        # 遍历字典中的每个小区
        for community, values in data_dict.items():
            # 为每个月创建一个元组
            values = values[0]
            for i, value in enumerate(values):
                if i < len(months):  # 确保索引不超出月份列表范围
                    # 只添加非零值到结果列表中
                    result.append((community, months[i], value))
    return result




def predict_final_result(data: dict, file_name: str, chain, save_lock):
    query = data['query']
    current_price = []
    if data['predict_history_SQL_result'] is not None:
        for result in data['predict_history_SQL_result']:
            if len(result) >= 2:
                if result[0] == '2023年12月' or result[1] == '2023年12月':
                    current_price.append(result)
    future_price = data['TimesNet_predict_price']
    history_price = data['predict_history_SQL_result']

    try:
        if future_price is not None:

            table_results = future_price
            if data['BERT_pred_query_type'] == "Judgment":
                response = chain.invoke({
                    "query": query,
                    "history_table_results": history_price,
                    "current_price": current_price,
                    "table_results": table_results
                })
            else:
                response = chain.invoke({
                    "query": query,
                    "history_table_results": history_price,
                    "table_results": table_results
                })
            answer = response.content
            if '</think>' in answer:
                answer = extract_content_after_think(answer)
            if '<Answer>:' in answer:
                answer = extract_content_after_Answer(answer)
            data['predict_answer'] = answer
        else:

            data['predict_answer'] = None

        saver = PredictResultStorage(file_name, save_lock)
        saver.set_data(data)
        saver.save_data()

    except Exception as e:

        print(f"Chain invocation failed: {str(e)}")



def predict_normalization_result(data: dict, file_name: str, chain, save_lock):
    query = data['query']
    llm_predict_answer = data['predict_answer']
    if data['predict_history_SQL_result'] != None:
        predict_final_answer = parse_list_string(data['predict_answer'])
        predict_list, true_list,  success = process_numerical_predict_true(predict_final_answer, data['answer'])
        if success != True:
            try:
                if data['query_type'] == "Judgment":
                    pass
                else:
                    response = chain.invoke({
                        "query": query,
                        "llm_predict_answer": llm_predict_answer,
                    })
                answer = response.content
                if '</think>' in answer:
                    answer = extract_content_after_think(answer)
                if '<Answer>:' in answer:
                    answer = extract_content_after_Answer(answer)
                data['predict_normalization_answer'] = answer
        
                # 无论是否有预测值，只要没有异常就保存数据
                saver = PredictResultStorage(file_name, save_lock)
                saver.set_data(data)
                saver.save_data()
        
            except Exception as e:
                print(f"API调用异常: {str(e)}")
        else:

            data['predict_normalization_answer'] = data['predict_answer']
            saver = PredictResultStorage(file_name, save_lock)
            saver.set_data(data)
            saver.save_data()
            
    else:

        data['predict_normalization_answer'] = data['predict_answer']
        saver = PredictResultStorage(file_name, save_lock)
        saver.set_data(data)
        saver.save_data()


def extract_content_after_think(response):
    keyword = '</think>'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]



def extract_content_after_Answer(response):
    keyword = '<Answer>:'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]


def list_to_dict(data_list):

    result = {}
    
    # 检查是否有3个元素的情况且中间元素相同（公共项）
    has_common_middle = False
    if all(len(item) == 3 for item in data_list):
        middle_items = [item[1] for item in data_list]
        has_common_middle = len(set(middle_items)) == 1  # 如果所有中间项都相同
    
    # 转换为字典
    for item in data_list:
        if has_common_middle:
            # 三元素且有公共项，使用第0位为键，第2位为值
            result[item[0]] = item[2]
        else:
            # 常规情况，使用第0位为键，最后一位为值
            result[item[0]] = item[-1]
            
    return result


def parse_list_string(list_string):
    try:
        return ast.literal_eval(list_string)
    except (SyntaxError, ValueError) as e:
        #print(f"解析错误: {e}")
        # 备用解析方法
        return fallback_parse(list_string)

# 备用解析方法
def fallback_parse(list_string):
    # 移除所有括号和空格，然后按逗号分割
    clean_string = list_string.replace('[', '').replace(']', '').strip()
    if clean_string:
        # 尝试将数值字符串转为浮点数
        try:
            return [[float(clean_string)]]
        except ValueError:
            return [[clean_string]]
    return [[]]

def extract_answer(s):
    return s.split('<Answer>:')[1]

def clean_data(data):
    result = []
    def parse_element(element):
        if isinstance(element, list):
            for item in element:
                parse_element(item)
        elif isinstance(element, str):
            try:
                # 尝试解析可能存在的嵌套结构
                parsed = ast.literal_eval(element)
                parse_element(parsed)
            except:
                # 清理字符串中的多余符号
                cleaned = element.strip('[]\'"')
                if cleaned:
                    result.append(cleaned)
    
    parse_element(data)
    return result

def calculate_metrics(true_sets, pred_sets):
    correct = 0  # 新增：记录完全匹配的样本数
    TP, FP, FN = 0, 0, 0
    
    for t, p in zip(true_sets, pred_sets):
        # 新增：判断集合是否完全相等
        if t == p:
            correct += 1
        # 原有指标计算保持不变
        TP += len(t & p)
        FP += len(p - t)
        FN += len(t - p)
    
    total_samples = len(true_sets)
    accuracy = correct / total_samples if total_samples > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"│ Accuracy      │ {accuracy*100:.2f} │")  # 新增行
    print(f"│ Precision (P) │ {precision*100:.2f} │")
    print(f"│ Recall (R)    │ {recall*100:.2f} │")
    print(f"│ F1 Score      │ {f1*100:.2f} │")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")



def extract_value_and_name(entry):
    value = None
    name = None
    if not isinstance(entry, list):
        return None, None
    for item in entry:
        if isinstance(item, (int, float)):
            if value is None:
                value = item
            else:  # 多个数值，无效条目
                return None, None
        elif isinstance(item, str):
            if name is None:
                name = item
            else:  # 多个字符串，无效条目
                return None, None
    return value, name

def extract_numbers(predict):
    numbers = []
    for item in predict:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, (int, float)):
                    numbers.append(sub)
        elif isinstance(item, (int, float)):
            numbers.append(item)
    return numbers

def process_numerical_predict_true(predict, true):
    # 1. 处理无输入情况
    if predict == '无输入' or predict == [['无输入']] or predict == None:
        return [], [], False
    # 2. 处理纯数字情况
    try:
        if isinstance(true, list) and len(true) == 1 and isinstance(true[0], list) and len(true[0]) == 1 and isinstance(true[0][0], (int, float)):
            true_num = true[0][0]
            predict_nums = extract_numbers(predict)
            if len(predict_nums) == 1:
                return [true_num], [predict_nums[0]], True
            else:
                return [], [], False
    except:
        pass
    
    # 3. 处理名称/年月-数字对情况
    try:
        # 解析true的结构
        true_pairs = []
        for item in true:
            if isinstance(item, list) and len(item) >= 2 and isinstance(item[0], str) and isinstance(item[1], (int, float)):
                true_pairs.append((item[0], item[1]))
        
        matched_true = []
        matched_pred = []
        
        # 遍历每个true条目查找匹配
        for name, t_value in true_pairs:
            for entry in predict:
                p_value, p_name = extract_value_and_name(entry)
                if p_name == name and p_value is not None and isinstance(p_value, (int, float)):
                    matched_true.append(t_value)
                    matched_pred.append(p_value)
                    break  # 找到匹配则停止搜索
        
        if len(matched_pred) > 0:
            return matched_true, matched_pred, True
        else:
            return [], [], False
    except:
        return [], [], False


def flatten(lst):
    """递归展开嵌套列表，返回一维字符串列表"""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(str(item))
    return result

def process_judgment_predict_true(predict, true):
    # 处理无输入情况
    if predict == '无输入' or predict == [['无输入']] or predict == None:
        return [], [], False
    
    # 展开真实值和预测值
    true_flat = flatten(true)
    predict_flat = flatten(predict)
    # 判断输出list中，任意元素为汉字str，就认为有答案有效，返回True
    for element in predict_flat:
        if isinstance(element, str):
            for c in element:
                if '\u4e00' <= c <= '\u9fff':
                    return predict_flat, true_flat, True
    return [], [], False              