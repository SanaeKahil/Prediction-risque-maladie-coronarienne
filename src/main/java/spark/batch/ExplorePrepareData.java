package spark.batch;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DoubleType;
import static org.apache.spark.sql.functions.*;

public class ExplorePrepareData {
    private SparkSession spark;
    private JavaSparkContext sc;
    private Dataset<Row> dataTrain,dataTest;
    private  SparkConf conf ;

    public ExplorePrepareData() {
        this.conf = new SparkConf().setAppName("CHDPredictionTask").setMaster("local[*]");
        this.sc = new JavaSparkContext(conf);
        this.spark = SparkSession.builder().appName("CHDPredictionTask").getOrCreate();
    }
    public void chargementData(String inputPath){
        this.dataTrain = this.spark.read().option("header", "true").option("inferSchema", true).csv(inputPath + "/train.csv");
        this.dataTrain=this.dataTrain.drop("id");
        this.dataTest = this.spark.read().option("header", "true").option("inferSchema", true).csv(inputPath + "/test.csv");
        this.dataTest=this.dataTest.drop("id");
    }

    public void traitementValeurManquantAbberants(){
        String[] columnNames ={"cigsPerDay","education","BPMeds","totChol","BMI","heartRate","glucose"};
        double modeValueTrain,modeValueTest;
        Dataset<Row> modeDataTrain,modeDataTest,dataTrainTmp,dataTestTmp;
        //dataSet sans valeur null pour calculer mod
        dataTrainTmp=this.dataTrain.na().drop().toDF();
        dataTestTmp=this.dataTest.na().drop().toDF();
        for(String colonne : columnNames){

            modeDataTrain = dataTrainTmp.na().drop(new String[]{colonne}).groupBy(col(colonne)).count().orderBy(col("count").desc());
            modeDataTest = dataTestTmp.na().drop(new String[]{colonne}).groupBy(col(colonne)).count().orderBy(col("count").desc());

            modeValueTrain = modeDataTrain.first().getAs(colonne);
            modeValueTest = modeDataTest.first().getAs(colonne);
            //valeur manquants
            this.dataTrain =this.dataTrain.na().fill(modeValueTrain, new String[]{colonne});
            this.dataTest =this.dataTest.na().fill(modeValueTest, new String[]{colonne});
            //valeurs abberant
            double[] f1_f3=this.calculF1_F3(this.dataTrain,colonne);
            this.dataTrain= this.dataTrain.filter(col(colonne).between(f1_f3[0],f1_f3[1]));
            f1_f3=this.calculF1_F3(this.dataTest,colonne);
            this.dataTest=this.dataTest.filter(col(colonne).between(f1_f3[0],f1_f3[1]));
        }
    }
    public void encodeColonne(){
        this.dataTrain=  this.dataTrain.withColumn("is_smoking", when(this.dataTrain .col("is_smoking").equalTo("YES"), 1).otherwise(0));
        this.dataTrain =  this.dataTrain.withColumn("Sex",when(this.dataTrain .col("sex").equalTo("F"), 1).otherwise(0));

        this.dataTest =  this.dataTest.withColumn("is_smoking", when(this.dataTest .col("is_smoking").equalTo("YES"), 1).otherwise(0));
        this.dataTest =  this.dataTest.withColumn("Sex",when(this.dataTest .col("sex").equalTo("F"), 1).otherwise(0));
    }
    public void convertionToDouble(){
        String[] numericCols = new String[]{"age", "cigsPerDay", "BPMeds", "prevalentStroke",
                "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"};
        for (String c : numericCols) {
            this.dataTrain =this.dataTrain.withColumn(c,col(c).cast("double"));
            this.dataTest =this.dataTest.withColumn(c,col(c).cast("double"));
        }
    }
    public void  assembleFeatures(){
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"Sex", "age", "is_smoking","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"})
                .setOutputCol("features");
        this.dataTrain = assembler.transform(this.dataTrain);
        this.dataTest=assembler.transform(this.dataTest);
    }
    public double[] calculF1_F3(Dataset<Row> data,String colonne){
        double[] quartiles =data.stat().approxQuantile(colonne, new double[]{0.25,0.75}, 0.05);
        double q1 = quartiles[0];
        double q3 = quartiles[1];
        double iqr = q3 - q1;
        double f1 = q1 - 1.5 * iqr;
        double f3 = q3 + 1.5 * iqr;
        return new double[]{f1,f3};
    }
    public void description(String outputDir){
        String[] columns = this.dataTrain.schema().fieldNames();
        Dataset<Row> numericData = this.dataTrain;
        for (String columnName : columns) {
            if (!(this.dataTrain.schema().apply(columnName).dataType() instanceof DoubleType)) {
                numericData=numericData.drop(columnName);
            }
        }
        Dataset<Row> desc = numericData.describe();
       // desc.coalesce(1).write().option("header", "true").csv(outputDir+"description");
    }
    public Dataset<Row> getDataTrain() {
        return dataTrain;
    }

    public Dataset<Row> getDataTest() {
        return dataTest;
    }

    public SparkSession getSpark() {
        return spark;
    }
}
