import pandas as pd
import os

def filter_csv_to_excel():
    # 定义文件夹路径
    input_folder = "C:/Users/admin/Desktop/input"
    output_folder = "C:/Users/admin/Desktop/output"

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取input文件夹下所有的CSV文件
    csv_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            csv_files.append(file)

    for i, csv_file in enumerate(csv_files, 1):
        # 原始文件路径
        original_path = os.path.join(input_folder, csv_file)

        # 读取CSV文件，指定第5行（索引4）作为表头
        df = pd.read_csv(original_path, header=4)  # 替换为你的实际文件路径

        # 遍历第一列的所有值，删除值不为"合计"的行
        # 假设第一列的列名可以通过 df.columns[0] 获取
        column_name = df.columns[0]  # 第一列的列名
        filtered_df = df[df[column_name] == "合计\t"]

        # 新文件名
        new_filename = f"file_output_{i}.xlsx"
        new_path = os.path.join(output_folder, new_filename)
        # 将筛选后的数据保存到新的Excel文件
        filtered_df.to_excel(new_path, index=False)

        print(f'第 {i} 轮')
        print(f"数据处理完成，已保存到: {os.path.abspath(new_path)}")
        print(f"原始数据行数: {len(df)}，筛选后行数: {len(filtered_df)}")

    print(f"\n处理完成！总共处理了 {len(csv_files)} 个CSV文件。")

if __name__ == "__main__":

    # 执行筛选和导出
    result = filter_csv_to_excel()