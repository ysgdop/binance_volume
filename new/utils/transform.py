
import duckdb
import pandas as pd

# ========== 修改这里 ==========
db_path = r"C:\C++  Project\py\Binance_data.duckdb"   # 你的 .duckdb 文件路径
excel_path = r"C:\C++  Project\py\Binance_data.xlsx"        # 输出的 Excel 路径
# =============================

con = duckdb.connect(db_path, read_only=True)

# 获取所有表名
tables = con.execute("SHOW TABLES").fetchall()

if not tables:
    print("数据库里没有表")
else:
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for (table_name,) in tables:
            print(f"正在导出表: {table_name}")
            df = con.execute(f'SELECT * FROM "{table_name}"').df()
            # Excel sheet 名称最长 31 个字符
            sheet_name = table_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n导出完成 → {excel_path}")

con.close()