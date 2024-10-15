package spark.batch;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import static org.apache.spark.sql.functions.col;

public class ClassModel {
 private LogisticRegression logisticRegression;
 private LogisticRegressionModel model;
 private Dataset<Row> prediction;
    private long nbr;
    private long nbr1;
    private double accuracy;

    public LogisticRegression getLogisticRegression() {
        return logisticRegression;
    }

    public long getNbr() {
        return nbr;
    }

    public long getNbr1() {
        return nbr1;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public ClassModel() {
        this.logisticRegression = new LogisticRegression()
                .setLabelCol("TenYearCHD")
                .setFeaturesCol("features")
                .setFitIntercept(true)
                .setMaxIter(100000);


    }
    public void entrainementAndSaveModel(Dataset<Row> data){
        this.model=logisticRegression.fit(data);
        try {
            this.model.save("src/main/resources/logisticRegressionModel");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }
    public void predicions(Dataset<Row> data,String outputDir,SparkSession spark){
        this.model=LogisticRegressionModel.load("src/main/resources/logisticRegressionModel");
        this.prediction= model.transform(data);
        // Sélectionner uniquement la colonne de prédiction
        Dataset<Row> output =this.prediction.select("prediction");
        this.nbr1= output.filter("prediction = 1.0").count();
        this.nbr= output.filter("prediction =0.0").count();
        output.write().mode(SaveMode.Overwrite).csv(outputDir);
    }
    public void evaluateModel(Dataset<Row> data) {
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];
        LogisticRegressionModel modelA;
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("TenYearCHD")
                .setFeaturesCol("features")
                .setFitIntercept(true)
                .setMaxIter(100000);
        modelA = lr.fit(trainingData);

        Dataset<Row> predictions = modelA.transform(testData);

        Dataset<Row> predictionsDouble = predictions.withColumn("TenYearCHD_double", col("TenYearCHD").cast("double"));
        predictionsDouble.show();
        Dataset<Row> correctPredictions = predictionsDouble.filter(col("prediction").equalTo(col("TenYearCHD_double")));
        long correctCount = correctPredictions.count();
        long totalCount = testData.count();
        this.accuracy=(double) correctCount / totalCount;
    }
}
