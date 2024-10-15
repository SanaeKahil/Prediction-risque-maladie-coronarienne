package spark.batch;
import com.google.common.base.Preconditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
public class Main{
    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class);
    private  ExplorePrepareData prepareData=new ExplorePrepareData();
    private ClassModel model=new ClassModel();
    public static void main(String[] args) {
        Preconditions.checkArgument(args.length > 1, "Please provide the path of input file and output dir as parameters.");
        new Main().run(args[0], args[1]);
    }
    public void run(String inputFilePath, String outputDir) {
        // Supprimer le répertoire de sortie s'il existe déjà

        FileSystem fs = null;
        /* preparation des donnes */
        //chargement data train + test
        prepareData.chargementData(inputFilePath);
        prepareData.convertionToDouble();
        prepareData.description(outputDir);

       //traitement valeurs manquants (train+test)
        prepareData.traitementValeurManquantAbberants();
        prepareData.encodeColonne();
        prepareData.convertionToDouble();
        prepareData.assembleFeatures();
        /*entrainement et sauvgarde model*/

        Path folderPath = new Path("src/main/resources/logisticRegressionModel");
        boolean modelExiste;
        try {
            fs = FileSystem.get(folderPath.toUri(), new org.apache.hadoop.conf.Configuration());
            modelExiste=fs.exists(folderPath) && fs.isDirectory(folderPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        if(modelExiste){ //model deja sauvgarder
            System.out.println("existe");
        }else
            model.entrainementAndSaveModel(prepareData.getDataTrain());

        //predictions
        model.predicions(prepareData.getDataTest(),outputDir,prepareData.getSpark());
        model.evaluateModel(prepareData.getDataTrain());
        System.out.println(" FIN" );
        prepareData.getSpark().stop();
        System.out.println("Accuracy: " +model.getAccuracy());
        System.out.println("nombre de personne avec une prediction 1 est  " +model.getNbr1());
        System.out.println("nombre de personne avec une prediction 0 est  " +model.getNbr());

    }
}