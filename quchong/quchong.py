import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def main():
    import pandas as pd

    # 加载数据
    df1 = pd.read_csv('D:\车险智能报价\训练数据\早20250101至今成交车行家自报价记录.csv')
    df2 = pd.read_csv('D:\车险智能报价\训练数据\早20250101至今未成交车行家自报价记录2.csv')
    # df3 = pd.read_csv('D:\车险智能报价\训练数据\车险报价全量20251116至今.csv')
    print("??????????????????")
    # 合并
    data = pd.concat([df1, df2], ignore_index=True)

    # 读取数据
    # data = pd.read_csv('D:\车险智能报价\训练数据\车险报价全量20251116至今.csv')
    # mapping = pd.read_csv('D:\车险智能报价\能源类型映射.csv')
    # print(mapping)

    # 打印处理前行数
    print(f"处理前行数: {len(data)}")

    # 将报价日期列转换为日期时间类型
    data['报价日期'] = pd.to_datetime(data['报价日期'])

    # 按车架号、报价日期的日期部分分组，保留每组中报价时间最大的一条
    # 首先提取日期部分用于分组
    data['报价日期_日期'] = data['报价日期'].dt.date

    # 按车架号和日期分组，保留报价时间最大的记录
    data_sorted = data.sort_values('报价日期', ascending=False)
    data_dedup = data_sorted.drop_duplicates(subset=['车架号', '报价日期_日期'], keep='first')

    # 删除临时列
    data_dedup = data_dedup.drop(columns=['报价日期_日期'])

    # 映射能源类型
    # 确保映射表的一级能源种类代码列名与数据列名一致
    # 假设映射表中列名为'一级能源种类代码'和'能源类型'
    # 如果列名不同可能需要调整
    # mapping_dict = dict(zip(mapping['一级能源种类代码'], mapping['能源类型']))
    # data_dedup['能源类型'] = data_dedup['一级能源种类代码'].map(mapping_dict)

    # 保存处理后的数据
    data_dedup.to_csv('D:\车险智能报价\训练数据\报价单去重早.csv', index=False, encoding='utf-8-sig')

    # 打印处理前后行数
    print(f"处理后行数: {len(data_dedup)}")
    print(f"去除了 {len(data) - len(data_dedup)} 条重复记录")


if __name__ == "__main__":
    # 执行主函数
    main()