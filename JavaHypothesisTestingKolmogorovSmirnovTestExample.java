package com.example.HypothesisTest;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.mllib.stat.test.KolmogorovSmirnovTestResult;

public class JavaHypothesisTestingKolmogorovSmirnovTestExample {
	public static void main(String[] args) throws NumberFormatException, IOException {
		SparkConf conf = new SparkConf().setAppName("JavaHypothesisTestingKolmogorovSmirnovTestExample").setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);

		String path = "input/haberman.data";	
		BufferedReader br = new BufferedReader(new FileReader(path));
		String line = null;
		List<Double> myList = new ArrayList<Double>();
		while ((line = br.readLine()) != null) {
			String[] tokens = line.split(",");
			myList.add(Double.parseDouble(tokens[0]));
			myList.add(Double.parseDouble(tokens[1]));
			myList.add(Double.parseDouble(tokens[2]));
			myList.add(Double.parseDouble(tokens[3])); 
			}
		
		Double [] list = myList.toArray(new Double[myList.size()]);		
		JavaDoubleRDD data = jsc.parallelizeDoubles(Arrays.asList(list));	
		//data.saveAsTextFile("output/data");

		KolmogorovSmirnovTestResult testResult = Statistics.kolmogorovSmirnovTest(data, "norm", 35.0, 1.5);
		// summary of the test including the p-value, test statistic, and null hypothesis if our p-value indicates significance, we can reject the null hypothesis
		System.out.println(testResult);

		jsc.stop();
	}
}
