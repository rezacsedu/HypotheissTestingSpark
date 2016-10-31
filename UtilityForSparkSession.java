package com.example.SparkSession;

import org.apache.spark.sql.SparkSession;
public class UtilityForSparkSession {
	public static SparkSession mySession() {
		SparkSession spark = SparkSession.builder().appName("JavaHypothesisTestingOnBreastCancerData").master("local[*]")
				.config("spark.sql.warehouse.dir", "E:/Exp/").getOrCreate();
		return spark;
	}
}
