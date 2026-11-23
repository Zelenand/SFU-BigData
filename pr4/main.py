import os
import re
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
# os.environ["HADOOP_HOME"] = "C:\\hadoop"
# os.environ["hadoop.home.dir"] = "C:\\hadoop"
# os.environ["PATH"] += os.pathsep + "C:\\hadoop\\bin"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

FILES = {
    "orders": "orders_clean",
    "order_items": "order_items_clean",
    "products": "products_clean",
    "customers": "customers_clean",
    "reviews": "reviews_clean",
}

spark = None
data = {}


def start_spark():
    global spark
    if spark is None:
        spark = (
            SparkSession.builder
            .appName("LR4_Olist_Spark")
            .master("local[*]")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.python.worker.faulthandler.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_data():
    start_spark()
    missing = []
    for key, filename in FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            missing.append(filename)
        else:
            data[key] = spark.read.parquet(str(path))

    if missing:
        print("\nНе найдены следующие файлы:")
        for filename in missing:
            print(f"  - data/{filename}")
        print("\nРаспакуйте очищенные данные из ЛР 3 в папку data.")
        return False

    print("\nДанные успешно загружены.")
    return True


def show_data_info():
    if not data and not load_data():
        return
    print("\n--- Информация о таблицах ---")
    for name, df in data.items():
        print(f"\n{name}: {df.count()} строк, {len(df.columns)} столбцов")
        print("Столбцы:", ", ".join(df.columns))


def get_orders_items():
    orders = data["orders"]
    items = data["order_items"]

    date_col = "order_purchase_timestamp"
    if date_col in orders.columns:
        orders = orders.withColumn(date_col, F.to_timestamp(date_col))

    return orders.join(items, "order_id", "inner")


def monthly_sales():
    if not data and not load_data():
        return

    joined = get_orders_items()
    date_col = "order_purchase_timestamp"

    result = (
        joined.withColumn("month", F.date_format(date_col, "yyyy-MM"))
        .groupBy("month")
        .agg(
            F.countDistinct("order_id").alias("orders_count"),
            F.sum("price").alias("revenue"),
            F.sum("freight_value").alias("freight_total"),
        )
        .orderBy("month")
    )

    print("\n--- Продажи по месяцам ---")
    result.show(50, truncate=False)

    pdf = result.toPandas()
    if not pdf.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(pdf["month"], pdf["revenue"], marker="o")
        plt.title("Выручка по месяцам")
        plt.xlabel("Месяц")
        plt.ylabel("Выручка")
        plt.xticks(rotation=45)
        plt.tight_layout()
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "monthly_revenue.png"
        plt.savefig(path)
        plt.show()
        print(f"График сохранён: {path}")


def category_analysis():
    if not data and not load_data():
        return

    products = data["products"]
    items = data["order_items"]

    category_col = next(
        (c for c in ["product_category_name_english", "product_category_name"]
         if c in products.columns),
        None
    )

    result = (
        items.join(products, "product_id", "left")
        .groupBy(category_col)
        .agg(
            F.count("*").alias("items_sold"),
            F.sum("price").alias("revenue")
        )
        .orderBy(F.desc("revenue"))
    )

    print("\n--- Продажи по категориям ---")
    result.show(20, truncate=False)

    pdf = result.limit(15).toPandas()
    if not pdf.empty:
        plt.figure(figsize=(12, 6))
        plt.bar(pdf[category_col].fillna("Не указано"), pdf["revenue"])
        plt.title("Топ-15 категорий по выручке")
        plt.xlabel("Категория")
        plt.ylabel("Выручка")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "category_revenue.png"
        plt.savefig(path)
        plt.show()
        print(f"График сохранён: {path}")


def regional_analysis():
    if not data and not load_data():
        return

    orders = data["orders"]
    customers = data["customers"]

    result = (
        orders.join(customers, "customer_id", "left")
        .groupBy("customer_state")
        .agg(F.countDistinct("order_id").alias("orders_count"))
        .orderBy(F.desc("orders_count"))
    )

    print("\n--- Количество заказов по штатам ---")
    result.show(30, truncate=False)

    pdf = result.toPandas()
    if not pdf.empty:
        plt.figure(figsize=(12, 6))
        plt.bar(pdf["customer_state"], pdf["orders_count"])
        plt.title("Количество заказов по штатам")
        plt.xlabel("Штат")
        plt.ylabel("Количество заказов")
        plt.xticks(rotation=45)
        plt.tight_layout()
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "orders_by_state.png"
        plt.savefig(path)
        plt.show()
        print(f"График сохранён: {path}")

    export_result(result, "orders_by_state")


def mapreduce_reviews():
    if not data and not load_data():
        return

    reviews = data["reviews"]
    text_col = "review_comment_message"

    if text_col not in reviews.columns:
        print(f"В таблице отзывов нет столбца {text_col}.")
        return

    text_rdd = (
        reviews.select(text_col)
        .filter(F.col(text_col).isNotNull())
        .filter(F.trim(F.col(text_col)) != "")
        .rdd.map(lambda row: row[0])
    )

    def mapper(text):
        if text is None:
            return []
        text = text.lower()
        text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
        return [(word, 1) for word in text.split() if len(word) > 2]

    mapped = text_rdd.flatMap(mapper)
    reduced = mapped.reduceByKey(lambda a, b: a + b)
    top_words = reduced.takeOrdered(20, key=lambda x: -x[1])

    print("\n--- MapReduce: 20 наиболее частых слов ---")
    for word, count in top_words:
        print(f"{word:25} {count}")

    if top_words:
        words = [x[0] for x in top_words]
        counts = [x[1] for x in top_words]
        plt.figure(figsize=(12, 6))
        plt.bar(words, counts)
        plt.title("Частота слов в отзывах покупателей")
        plt.xlabel("Слово")
        plt.ylabel("Количество")
        plt.xticks(rotation=45)
        plt.tight_layout()
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "review_word_frequency.png"
        plt.savefig(path)
        plt.show()
        print(f"График сохранён: {path}")

    word_df = spark.createDataFrame(top_words, ["word", "frequency"])
    export_result(word_df, "word_frequency")


def export_result(df, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / f"{name}_csv"
    parquet_path = OUTPUT_DIR / f"{name}_parquet"

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(csv_path))
    df.write.mode("overwrite").parquet(str(parquet_path))

    print(f"Результаты сохранены:\n  CSV: {csv_path}\n  Parquet: {parquet_path}")


def export_all():
    if not data and not load_data():
        return

    export_result(
        get_orders_items()
        .withColumn("month", F.date_format("order_purchase_timestamp", "yyyy-MM"))
        .groupBy("month")
        .agg(
            F.countDistinct("order_id").alias("orders_count"),
            F.sum("price").alias("revenue"),
            F.sum("freight_value").alias("freight_total"),
        )
        .orderBy("month"),
        "monthly_sales"
    )

    category_col = next(
        (c for c in ["product_category_name_english", "product_category_name"]
         if c in data["products"].columns),
        None
    )
    if category_col:
        category_result = (
            data["order_items"].join(data["products"], "product_id", "left")
            .groupBy(category_col)
            .agg(
                F.count("*").alias("items_sold"),
                F.sum("price").alias("revenue")
            )
            .orderBy(F.desc("revenue"))
        )
        export_result(category_result, "category_sales")

    print("\nЭкспорт завершён.")


def menu():
    while True:
        print("\n" + "=" * 45)
        print(" АНАЛИТИКА ДАННЫХ E-COMMERCE")
        print(" Apache Spark | Лабораторная работа № 4")
        print("=" * 45)
        print("1. Просмотр загруженных данных")
        print("2. Анализ продаж по месяцам")
        print("3. Анализ продаж по категориям")
        print("4. Анализ заказов по регионам")
        print("5. MapReduce: анализ текстовых отзывов")
        print("6. Сохранение аналитических результатов")
        print("0. Выход")

        choice = input("\nВыберите действие: ").strip()

        if choice == "1":
            show_data_info()
        elif choice == "2":
            monthly_sales()
        elif choice == "3":
            category_analysis()
        
        elif choice == "4":
            regional_analysis()
        elif choice == "5":
            mapreduce_reviews()
        elif choice == "6":
            export_all()
        elif choice == "0":
            print("Работа приложения завершена.")
            break
        else:
            print("Некорректный выбор. Попробуйте ещё раз.")


if __name__ == "__main__":
    try:
        load_data()
        menu()
    finally:
        if spark is not None:
            spark.stop()
