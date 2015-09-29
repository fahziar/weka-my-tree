package tugas;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.MyId3;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.util.Random;

/**
 * Created by fahziar on 29/09/2015.
 */
public class Main {
    private static final String DATASET_WEATHER = "weather.nominal.arff";
    private static final String DATASET_WEATHER_NUM = "weather.numeric.arff";
    private static final String DATASET_IRIS = "iris.arff";

    private static Instances loadDataset(String location) throws Exception {
        ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(location);
        Instances data = dataSource.getDataSet();

        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes() - 1);
        }

        return data;
    }

    private static Instances resample(Instances dataset) throws Exception {
        Resample resample = new Resample();

        String[] options = new String[3];
        options[0] = "-B 0.0";
        options[1] = "-S 1";
        options[2] = "-Z 100.0";

        resample.setOptions(options);
        resample.setInputFormat(dataset);
        return Filter.useFilter(dataset, resample);

    }

    public static void main(String[] args){
        Instances irisDs;
        Instances weatherDs;
        Instances numWeatherDs;

        //Load dataset
        try {
            System.out.println("Loading dataset dataset");
            weatherDs = loadDataset(DATASET_WEATHER);
            numWeatherDs = loadDataset(DATASET_WEATHER_NUM);
            irisDs = loadDataset(DATASET_IRIS);
        } catch (Exception e){
            e.printStackTrace();
            System.out.println("Failed to load dataset");
            return;
        }

        //Resample dataset
        try {
            System.out.println("Filtering dataset");
            weatherDs = resample(weatherDs);
            numWeatherDs = resample(numWeatherDs);
            irisDs = resample(irisDs);
        } catch (Exception e){
            e.printStackTrace();
            System.out.println("Failted to filter dataset");
            return;
        }

        //Build classifier
        System.out.println("Building classifier");

        MyId3 myId3Weather = new MyId3();
        MyId3 myId3Iris = new MyId3();

        Id3 id3Weather = new Id3();
        Id3 id3Iris = new Id3();

        J48 j48Weather = new J48();
        J48 j48NumWeather = new J48();
        J48 j48Iris = new J48();

        NaiveBayes naiveWeather = new NaiveBayes();
        NaiveBayes naiveNumWeather = new NaiveBayes();
        NaiveBayes naiveIris = new NaiveBayes();


        try {

            myId3Weather.buildClassifier(weatherDs);
//            myId3Iris.buildClassifier(irisDs);

            id3Weather.buildClassifier(weatherDs);
            //id3Iris.buildClassifier(irisDs);

            j48Weather.buildClassifier(weatherDs);
            j48NumWeather.buildClassifier(numWeatherDs);
            j48Iris.buildClassifier(irisDs);

            naiveWeather.buildClassifier(weatherDs);
            naiveNumWeather.buildClassifier(numWeatherDs);
            naiveIris.buildClassifier(irisDs);

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Failed to build classifier");
            return;
        }

        //10 cross validation dataset
        try {
            Evaluation weatherEvalMyId3 = new Evaluation(weatherDs);
            Evaluation weatherEvalId3 = new Evaluation(weatherDs);
            Evaluation weatherEvalJ48 = new Evaluation(weatherDs);
            Evaluation weatherNumEval = new Evaluation(numWeatherDs);
            Evaluation irisEval = new Evaluation(irisDs);

            //My ID3
            weatherEvalMyId3.crossValidateModel(myId3Weather, weatherDs, 10, new Random(1));
            System.out.println(weatherEvalMyId3.toSummaryString("\nMyID3 Weathenr Nominal Results\n======\n", false));
//            irisEval.crossValidateModel(myId3Iris, irisDs, 10, new Random(1));
//            System.out.println(irisEval.toSummaryString("\nMyID3 iris Results\n======\n", false));

            //ID3
            weatherEvalId3.crossValidateModel(id3Weather, weatherDs, 10, new Random(1));
            System.out.println(weatherEvalId3.toSummaryString("\nID3 Weathenr Nominal Results\n======\n", false));
//            irisEval.crossValidateModel(id3Iris, irisDs, 10, new Random(1));
//            System.out.println(irisEval.toSummaryString("\nID3 iris Results\n======\n", false));

            //J48
            weatherEvalJ48.crossValidateModel(j48Weather, weatherDs, 10, new Random(1));
            System.out.println(weatherEvalJ48.toSummaryString("\nJ48 Weather Nominal Results\n======\n", false));
            weatherNumEval.crossValidateModel(j48NumWeather, numWeatherDs, 10, new Random(1));
            System.out.println(weatherNumEval.toSummaryString("\nJ48 Weather Numeric Results\n======\n", false));
            irisEval.crossValidateModel(j48Iris, irisDs, 10, new Random(1));
            System.out.println(irisEval.toSummaryString("\nJ48 iris Results\n======\n", false));


        } catch (Exception e){
            e.printStackTrace();
            System.out.println("Failed to evaluate dataset");
            return;
        }

        //Save model
        try {
            System.out.println("Saving model");
            SerializationHelper.write("myId3Weather.model", myId3Weather);
//            SerializationHelper.write("myId3Iris.model", myId3Iris);

            SerializationHelper.write("Id3Weather.model", id3Weather);
//            SerializationHelper.write("Id3Iris.model", id3Iris);

            SerializationHelper.write("J48Weather.model", j48Weather);
            SerializationHelper.write("J48NumWeather.model", j48NumWeather);
            SerializationHelper.write("J48Iris.model", j48Iris);

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Failed to save model");
        }
    }
}
