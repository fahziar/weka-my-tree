package weka.classifiers.trees.my;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

/**
 * Created by Alvin Natawiguna on 9/29/2015.
 */
public class C45ModelSelection {
    /** Minimum number of objects in interval. */
    private int minObjectsInInterval;

    /** All the training data */
    private Instances trainData; //

    /**
     * Initializes the split selection method with the given parameters.
     *
     * @param minNoObj minimum number of instances that have to occur in at least two
     * subsets induced by split
     * @param allData FULL training dataset (necessary for
     * selection of split points).
     */
    public C45ModelSelection(int minNoObj, Instances allData) {
        minObjectsInInterval = minNoObj;
        trainData = allData;
    }

    /**
     * Sets reference to training data to null.
     */
    public void cleanup() {
        trainData = null;
    }

    /**
     * Selects C4.5-type split for the given dataset.
     */
    public final C45SplitModel selectModel(Instances data){
        C45SplitModel bestModel = null;

        double averageInfoGain = 0;
        int validModels = 0;
        boolean multiVal = true;

        try {

            // Check if all Instances belong to one class or if not
            // enough Instances to split.
            C45Distribution checkDistribution = new C45Distribution(data);
            C45NoSplitModel noSplitModel = new C45NoSplitModel(checkDistribution);

            if (Utils.sm(checkDistribution.total(), 2 * minObjectsInInterval) ||
                    Utils.eq(checkDistribution.total(),
                            checkDistribution.perClass(checkDistribution.maxClass())))
                return noSplitModel;

            // Check if all attributes are nominal and have a
            // lot of values.
            if (trainData != null) {
                Enumeration enu = data.enumerateAttributes();
                while (enu.hasMoreElements()) {
                    Attribute attribute = (Attribute) enu.nextElement();
                    if ((attribute.isNumeric()) ||
                            (Utils.sm((double)attribute.numValues(),
                                    (0.3*(double) trainData.numInstances()))))
                    {
                        multiVal = false;
                        break;
                    }
                }
            }

            C45SplitModel []currentModel = new C45SplitModel[data.numAttributes()];
            double sumOfWeights = data.sumOfWeights();

            // For each attribute.
            for (int i = 0; i < data.numAttributes(); i++){

                // Apart from class attribute.
                if (i != (data).classIndex()) {

                    // Get models for current attribute.
                    currentModel[i] = new C45SplitModel(i, minObjectsInInterval, sumOfWeights);
                    currentModel[i].buildClassifier(data);

                    // Check if useful split for current attribute
                    // exists and check for enumerated attributes with
                    // a lot of values.
                    if (currentModel[i].checkModel())
                        if (trainData != null) {
                            if ((data.attribute(i).isNumeric()) ||
                                    (multiVal || Utils.sm((double)data.attribute(i).numValues(),
                                            (0.3*(double) trainData.numInstances())))){
                                averageInfoGain += currentModel[i].getInfoGain();
                                validModels++;
                            }
                        } else {
                            averageInfoGain += currentModel[i].getInfoGain();
                            validModels++;
                        }
                }else
                    currentModel[i] = null;
            }

            // Check if any useful split was found.
            if (validModels == 0) {
                return noSplitModel;
            }

            averageInfoGain /= (double) validModels;

            // Find "best" attribute to split on.
            double minResult = 0;
            for (int i = 0; i < data.numAttributes(); i++){
                if ((i != (data).classIndex()) &&
                        (currentModel[i].checkModel()))

                    // Use 1E-3 here to get a closer approximation to the original
                    // implementation.
                    if ((currentModel[i].getInfoGain() >= (averageInfoGain - 1E-3)) &&
                            Utils.gr(currentModel[i].getGainRatio(), minResult))
                    {
                        bestModel = currentModel[i];
                        minResult = currentModel[i].getGainRatio();
                    }
            }

            // Check if useful split was found.
            if (Utils.eq(minResult,0))
                return noSplitModel;

            // Add all Instances with unknown values for the corresponding
            // attribute to the distribution for the model, so that
            // the complete distribution is stored with the model.
            assert bestModel != null;
            bestModel.getDistribution().addInstWithUnknown(data, bestModel.getAttributeIndex());

            // Set the split point analogue to C45 if attribute numeric.
            if (trainData != null) {
                bestModel.setSplitPoint(trainData);
            }

            return bestModel;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }

    /**
     * Selects C4.5-type split for the given dataset.
     */
    public final C45SplitModel selectModel(Instances train, Instances test) {
        return selectModel(train);
    }
}
