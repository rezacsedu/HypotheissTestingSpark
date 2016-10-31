package com.example.HypothesisTest;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.mllib.stat.test.ChiSqTestResult;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import com.example.SparkSession.UtilityForSparkSession;


// Note: According to Source: R.A. Fisher and F. Yates, Statistical Tables for Biological Agricultural and Medical Research, 6th ed., 
//Table IV, Oliver & Boyd, Ltd., Edinburgh, by permission of the authors and publishers. Here goes two rule of thumbs:
//i) If the p value for the calculated 2 is p > 0.05, accept your hypothesis. 
   //'The deviation is small enough that chance alone accounts for it. A p value of 0.6, for example, means that there is a 60% probability that any deviation from expected is due to chance only. This is within the range of acceptable deviation.
//ii)If the p value for the calculated 2 is p < 0.05, reject your hypothesis, and conclude that some factor other than chance is operating for the deviation to be so great. For example, a p value of 0.01 means that there 
     //is only a 1% chance that this deviation is due to chance alone. Therefore, other factors must be involved.



public class JavaHypothesisTestingOnBreastCancerData {
	//Create Spark context
    static SparkSession spark = UtilityForSparkSession.mySession();
	static String path = "input/wpbc.data";
	
	//This method creates dense vcector from the breast cancer diagnosis data. 
	//Since the the first two records are patient id and diagnosis label.
	//Note we ignore first two column for the simplicy.  
	public static Vector myVector() throws NumberFormatException, IOException {		
		BufferedReader br = new BufferedReader(new FileReader(path));
		String line = null;
		Vector v = null;
		while ((line = br.readLine()) != null) {
			String[] tokens = line.split(",");
			double[] features = new double[30];
			for (int i = 2; i < features.length; i++) {
				features[i-2] = Double.parseDouble(tokens[i]);
			}
			v = new DenseVector(features);
		}
		return v;
	}

	public static void main(String[] args) throws NumberFormatException, IOException {
		//Collect the vectors that we created using the myVector() method
		Vector v = myVector();
		
		// compute the goodness of fit. If a second vector to test against is not supplied as a parameter, the test runs against a uniform distribution.
		ChiSqTestResult goodnessOfFitTestResult = Statistics.chiSqTest(v);
		// summary of the test including the p-value, degrees of freedom, test
		// statistic, the method used, and the null hypothesis.
		System.out.println(goodnessOfFitTestResult + "\n");

		// Create a contingency matrix ((1.0, 3.0, 5.0, 2.0), (4.0, 6.0, 1.0, 3.5), (6.9, 8.9, 10.5, 12.6))
		Matrix mat = Matrices.dense(4, 3, new double[] { 1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 1.0, 3.5, 6.9, 8.9, 10.5, 12.6});
		// conduct Pearson's independence test on the input contingency matrix
		ChiSqTestResult independenceTestResult = Statistics.chiSqTest(mat);
		// summary of the test including the p-value, degrees of freedom...
		System.out.println(independenceTestResult + "\n");
		
		// Create an RDD of labeled points
		RDD<String> lines = spark.sparkContext().textFile(path, 2);		
		JavaRDD<LabeledPoint> linesRDD = lines.toJavaRDD().map(new Function<String, LabeledPoint>() {
					public LabeledPoint call(String lines) {
						String[] tokens = lines.split(",");
						double[] features = new double[30];
						for (int i = 2; i < features.length; i++) {
							features[i - 2] = Double.parseDouble(tokens[i]);
						}
						Vector v = new DenseVector(features);
						if (tokens[1].equals("R")) {
							return new LabeledPoint(1.0, v); // recurrent
						} else {
							return new LabeledPoint(0.0, v); // non-recurrent
						}
					}
				});

		// The contingency table is constructed from the raw (feature, label) pairs and used to conduct the independence test. Returns an array containing the
		// ChiSquaredTestResult for every feature against the label.
		ChiSqTestResult[] featureTestResults = Statistics.chiSqTest(linesRDD.rdd());
		int i = 1;
		for (ChiSqTestResult result : featureTestResults) {
			System.out.println("Column " + i + ":");
			System.out.println(result + "\n"); // summary of the test
			i++;
		}
		
		//Finally, close the spark session
		spark.stop();
		System.exit(0);
	}
}
