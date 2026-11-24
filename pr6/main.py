import re
from pathlib import Path
import os
import sys
import json
from datetime import datetime

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop.home.dir"] = "C:\\hadoop"
os.environ["PATH"] += os.pathsep + "C:\\hadoop\\bin"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data" / "output"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "olist-orders")
STREAM_OUTPUT_DIR = OUTPUT_DIR / "kafka_streaming"
STREAM_CHECKPOINT_DIR = OUTPUT_DIR / "kafka_checkpoint"
MODEL_DIR = OUTPUT_DIR / "sentiment_pipeline_model"

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
        # Подбираем Scala-версию Kafka-коннектора под установленный PySpark.
        import pyspark
        spark_version = pyspark.__version__
        scala_suffix = "2.13" if spark_version.startswith("4.") else "2.12"
        kafka_package = (
            f"org.apache.spark:spark-sql-kafka-0-10_{scala_suffix}:{spark_version}"
        )

        spark = (
            SparkSession.builder
            .appName("Olist_LR6_Analytics")
            .master("local[*]")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.python.worker.faulthandler.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .config("spark.jars.packages", kafka_package)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        print(f"Kafka-коннектор: {kafka_package}")
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
        print("\nНе найдены следующие Parquet-папки/файлы:")
        for filename in missing:
            print(f"  - data/{filename}")
        print("Скопируйте очищенные данные из ЛР 3 в папку data.")
        return False

    print("\nОчищенные данные из ЛР 3 успешно загружены.")
    return True


def ensure_data():
    return bool(data) or load_data()


def show_data_info():
    if not ensure_data():
        return
    print("\n--- Информация о таблицах ---")
    for name, df in data.items():
        print(f"\n{name}: {df.count()} строк, {len(df.columns)} столбцов")
        print("Столбцы:", ", ".join(df.columns))


def get_orders_items():
    orders = data["orders"]
    items = data["order_items"]
    if "order_purchase_timestamp" in orders.columns:
        orders = orders.withColumn(
            "order_purchase_timestamp",
            F.to_timestamp("order_purchase_timestamp")
        )
    return orders.join(items, "order_id", "inner")


def save_plot(fig, filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"График сохранён: {path}")


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
    if not ensure_data():
        return
    reviews = data["reviews"]
    text_col = "review_comment_message"

    text_rdd = (
        reviews.select(text_col)
        .filter(F.col(text_col).isNotNull())
        .filter(F.trim(F.col(text_col)) != "")
        .rdd.map(lambda row: row[0])
    )

    def mapper(text):
        normalized = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text.lower())
        return [(word, 1) for word in normalized.split() if len(word) > 2]

    top_words = (
        text_rdd.flatMap(mapper)
        .reduceByKey(lambda a, b: a + b)
        .takeOrdered(20, key=lambda x: -x[1])
    )

    print("\n--- MapReduce: 20 наиболее частых слов ---")
    for word, count in top_words:
        print(f"{word:25} {count}")

    if top_words:
        words, counts = zip(*top_words)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(words, counts)
        ax.set(title="Частота слов в отзывах", xlabel="Слово", ylabel="Количество")
        ax.tick_params(axis="x", rotation=45)
        save_plot(fig, "review_word_frequency.png")

    export_result(
        spark.createDataFrame(top_words, ["word", "frequency"]),
        "word_frequency"
    )


def build_ml_dataset():
    reviews, orders, items = data["reviews"], data["orders"], data["order_items"]
    required_reviews = {"order_id", "review_score", "review_comment_message"}
    required_orders = {"order_id", "order_purchase_timestamp"}
    required_items = {"order_id", "price"}

    for label, df, required in [
        ("reviews", reviews, required_reviews),
        ("orders", orders, required_orders),
        ("order_items", items, required_items)
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"В таблице {label} отсутствуют столбцы: {sorted(missing)}")

    items_agg = items.groupBy("order_id").agg(
        F.sum("price").alias("order_value"),
        F.count("*").alias("items_count")
    )

    order_cols = ["order_id", "order_purchase_timestamp"]
    if "order_delivered_customer_date" in orders.columns:
        order_cols.append("order_delivered_customer_date")

    joined = (
        reviews.select("order_id", "review_score", "review_comment_message")
        .join(orders.select(*order_cols), "order_id", "left")
        .join(items_agg, "order_id", "left")
    )

    if "order_delivered_customer_date" in joined.columns:
        joined = joined.withColumn(
            "delivery_days",
            F.when(
                F.col("order_delivered_customer_date").isNotNull()
                & F.col("order_purchase_timestamp").isNotNull(),
                (F.unix_timestamp("order_delivered_customer_date")
                 - F.unix_timestamp("order_purchase_timestamp")) / 86400.0
            )
        )
    else:
        joined = joined.withColumn("delivery_days", F.lit(None).cast("double"))

    dataset = (
        joined
        .withColumn("review_score", F.col("review_score").cast("double"))
        .filter(F.col("review_score").isin(1.0, 2.0, 4.0, 5.0))
        .filter(F.col("review_comment_message").isNotNull())
        .filter(F.length(F.trim("review_comment_message")) > 0)
        .withColumn("label", F.when(F.col("review_score") >= 4, 1.0).otherwise(0.0))
        .withColumn("order_value", F.coalesce(F.col("order_value"), F.lit(0.0)).cast("double"))
        .withColumn("items_count", F.coalesce(F.col("items_count"), F.lit(0)).cast("double"))
        .withColumn("delivery_days", F.coalesce(F.col("delivery_days"), F.lit(0.0)).cast("double"))
        .filter(F.col("delivery_days") >= 0)
    )
    return dataset


def machine_learning():
    if not ensure_data():
        return

    print("\nПодготовка набора данных для классификации отзывов...")
    try:
        dataset = build_ml_dataset().cache()
        total = dataset.count()
        if total < 30:
            print(f"Недостаточно записей для обучения: {total}. Нужно хотя бы 30.")
            dataset.unpersist()
            return

        print(f"Записей для обучения: {total}")
        dataset.groupBy("label").count().show()

        train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=42)
        train_count, test_count = train_data.count(), test_data.count()
        if train_count == 0 or test_count == 0:
            print("Не удалось сформировать обучающую и тестовую выборки.")
            dataset.unpersist()
            return

        tokenizer = RegexTokenizer(
            inputCol="review_comment_message",
            outputCol="tokens",
            pattern=r"\W+",
            minTokenLength=2,
            toLowercase=True
        )
        stop_words = [
            "de", "a", "o", "que", "e", "do", "da", "em", "um", "uma",
            "para", "com", "não", "nao", "no", "por", "os", "as", "dos",
            "das", "ao", "mais", "muito", "foi", "ser", "se", "me", "eu",
            "como", "já", "ja", "mas", "ou", "sua", "seu"
        ]
        remover = StopWordsRemover(
            inputCol="tokens", outputCol="filtered_tokens",
            stopWords=stop_words
        )
        vectorizer = CountVectorizer(
            inputCol="filtered_tokens", outputCol="raw_features",
            vocabSize=5000, minDF=2
        )
        idf = IDF(inputCol="raw_features", outputCol="text_features")
        assembler = VectorAssembler(
            inputCols=["text_features", "order_value", "items_count", "delivery_days"],
            outputCol="features"
        )
        classifier = LogisticRegression(
            featuresCol="features", labelCol="label",
            maxIter=30, regParam=0.01
        )
        pipeline = Pipeline(stages=[
            tokenizer, remover, vectorizer, idf, assembler, classifier
        ])

        print("Обучение Logistic Regression...")
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data).cache()

        accuracy = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction",
            metricName="accuracy"
        ).evaluate(predictions)
        f1 = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction",
            metricName="f1"
        ).evaluate(predictions)

        try:
            auc = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            ).evaluate(predictions)
        except Exception as exc:
            auc = None
            print("Не удалось рассчитать ROC-AUC:", exc)

        print("\n--- Результаты оценки модели ---")
        print(f"Обучающая выборка: {train_count}")
        print(f"Тестовая выборка: {test_count}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}" if auc is not None else "ROC-AUC: не рассчитан")

        print("\nМатрица ошибок (label — фактический класс, prediction — прогноз):")
        confusion = predictions.groupBy("label", "prediction").count().orderBy("label", "prediction")
        confusion.show()

        OUTPUT_DIR.mkdir(exist_ok=True)
        model.write().overwrite().save(str(MODEL_DIR))
        pred_path = OUTPUT_DIR / "lr5_predictions"
        predictions.select(
            "order_id", "review_score", "label", "prediction", "probability"
        ).write.mode("overwrite").parquet(str(pred_path))

        metric_rows = [(float(accuracy), float(f1), float(auc) if auc is not None else None)]
        metrics_df = spark.createDataFrame(
            metric_rows, ["accuracy", "f1_score", "roc_auc"]
        )
        metrics_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
            str(OUTPUT_DIR / "lr5_metrics_csv")
        )

        print(f"\nМодель сохранена: {MODEL_DIR}")
        print(f"Предсказания сохранены: {pred_path}")
        print("Метрики сохранены в output/lr5_metrics_csv")

        predictions.unpersist()
        dataset.unpersist()

    except Exception as exc:
        print("\nОшибка при выполнении машинного обучения:")
        print(exc)
        print("Проверьте названия столбцов и структуру Parquet-таблиц.")


def load_saved_model_predict():
    if not ensure_data():
        return
    if not MODEL_DIR.exists():
        print("Сохранённая модель не найдена. Сначала выполните пункт обучения.")
        return
    try:
        model = PipelineModel.load(str(MODEL_DIR))
        dataset = build_ml_dataset()
        result = model.transform(dataset)
        result.select("order_id", "review_score", "label", "prediction").show(20)
        print("Предсказание выполнено с помощью сохранённой модели.")
    except Exception as exc:
        print("Не удалось загрузить модель или выполнить предсказание:", exc)




def send_orders_to_kafka():
    """Отправляет агрегированные события заказов из Parquet в Kafka."""
    if not ensure_data():
        return

    from kafka import KafkaProducer

    try:
        items_agg = data["order_items"].groupBy("order_id").agg(
            F.sum("price").alias("order_value"),
            F.count("*").alias("items_count")
        )
        orders = data["orders"]
        if "order_purchase_timestamp" not in orders.columns:
            print("В таблице orders отсутствует order_purchase_timestamp.")
            return

        events = (
            orders.select("order_id", "order_purchase_timestamp")
            .join(items_agg, "order_id", "inner")
            .withColumn("order_timestamp", F.to_timestamp("order_purchase_timestamp"))
            .filter(F.col("order_timestamp").isNotNull())
            .filter(F.col("order_value").isNotNull() & (F.col("order_value") > 0))
            .filter(F.col("items_count") > 0)
            .select("order_id", "order_timestamp", "order_value", "items_count")
            .orderBy("order_timestamp")
        )

        total = events.count()
        print(f"Подготовлено событий заказов: {total}")
        limit_text = input("Сколько событий отправить? (Enter — все): ").strip()
        if limit_text:
            try:
                limit = int(limit_text)
                if limit <= 0:
                    print("Количество должно быть больше нуля.")
                    return
                events = events.limit(limit)
            except ValueError:
                print("Введите целое число.")
                return

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
            retries=3
        )

        sent = 0
        for row in events.toLocalIterator():
            event = {
                "order_id": str(row["order_id"]),
                "order_timestamp": row["order_timestamp"].isoformat(sep=" "),
                "order_value": float(row["order_value"]),
                "items_count": int(row["items_count"])
            }
            producer.send(KAFKA_TOPIC, value=event)
            sent += 1
            if sent % 100 == 0:
                print(f"Отправлено событий: {sent}")

        producer.flush()
        producer.close()
        print(f"Готово. В Kafka отправлено событий: {sent}")
        print(f"Адрес Kafka: {KAFKA_BOOTSTRAP_SERVERS}, топик: {KAFKA_TOPIC}")

    except Exception as exc:
        print("\nНе удалось отправить события в Kafka:")
        print(exc)
        print("Проверьте, что Kafka запущена и доступна по адресу localhost:9092.")


def start_kafka_streaming():
    """Читает события из Kafka, агрегирует их по минутным окнам и сохраняет Parquet."""
    if not ensure_data():
        return

    try:
        schema = StructType([
            StructField("order_id", StringType(), True),
            StructField("order_timestamp", StringType(), True),
            StructField("order_value", DoubleType(), True),
            StructField("items_count", IntegerType(), True)
        ])

        kafka_stream = (
            spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
            .option("subscribe", KAFKA_TOPIC)
            .option("startingOffsets", "earliest")
            .load()
        )

        parsed = (
            kafka_stream
            .select(F.from_json(F.col("value").cast("string"), schema).alias("event"))
            .select("event.*")
            .withColumn("order_timestamp", F.to_timestamp("order_timestamp"))
            .filter(F.col("order_id").isNotNull())
            .filter(F.col("order_timestamp").isNotNull())
            .filter(F.col("order_value").isNotNull() & (F.col("order_value") > 0))
            .filter(F.col("items_count").isNotNull() & (F.col("items_count") > 0))
        )

        aggregated = (
            parsed
            .withWatermark("order_timestamp", "2 minutes")
            .groupBy(F.window("order_timestamp", "1 minute"))
            .agg(
                F.count("order_id").alias("orders_count"),
                F.sum("order_value").alias("revenue"),
                F.avg("order_value").alias("average_order_value"),
                F.sum("items_count").alias("items_count")
            )
            .select(
                F.col("window.start").alias("window_start"),
                F.col("window.end").alias("window_end"),
                "orders_count", "revenue", "average_order_value", "items_count"
            )
        )

        STREAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        STREAM_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        print("Потоковая обработка запущена. Ожидание до 60 секунд...")
        print(f"Топик: {KAFKA_TOPIC}")
        print(f"Результаты Parquet: {STREAM_OUTPUT_DIR}")
        print("Для остановки нажмите Ctrl+C.\n")

        query = (
            aggregated.writeStream
            .format("parquet")
            .outputMode("append")
            .option("path", str(STREAM_OUTPUT_DIR))
            .option("checkpointLocation", str(STREAM_CHECKPOINT_DIR))
            .trigger(processingTime="5 seconds")
            .start()
        )
        query.awaitTermination(60)
        query.stop()

    except KeyboardInterrupt:
        print("\nПотоковая обработка остановлена пользователем.")
    except Exception as exc:
        print("\nОшибка потоковой обработки Kafka:")
        print(exc)
        print("Проверьте запуск Kafka, топик и настройки подключения.")


def show_streaming_results():
    if not STREAM_OUTPUT_DIR.exists() or not any(STREAM_OUTPUT_DIR.glob("*.parquet")):
        print("Пока нет сохранённых результатов. Сначала запустите потоковую обработку.")
        return
    try:
        result = spark.read.parquet(str(STREAM_OUTPUT_DIR))
        print("\n--- Результаты Kafka Streaming по минутным окнам ---")
        result.orderBy("window_start").show(100, truncate=False)
    except Exception as exc:
        print("Не удалось прочитать результаты:", exc)



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
        print(" Apache Spark")
        print("=" * 45)
        print("1. Просмотр загруженных данных")
        print("2. Анализ продаж по месяцам")
        print("3. Анализ продаж по категориям")
        print("4. Анализ заказов по регионам")
        print("5. MapReduce: анализ текстовых отзывов")
        print("6. Сохранение аналитических результатов")
        print("7. Машинное обучение: классификация отзывов")
        print("8. Предсказание с помощью сохранённой ML-модели")
        print("9. Kafka Producer: отправка заказов")
        print("10. Spark Streaming: обработка заказов из Kafka")
        print("11. Просмотр результатов потоковой обработки")
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
        elif choice == "7":
            machine_learning()
        elif choice == "8":
            load_saved_model_predict()
        elif choice == "9":
            send_orders_to_kafka()
        elif choice == "10":
            start_kafka_streaming()
        elif choice == "11":
            show_streaming_results()
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