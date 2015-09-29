package weka.classifiers.trees.my;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

/**
 * Created by Alvin Natawiguna on 9/28/2015.
 */
public class C45SplitModel {

    /** Desired number of branches. */
    private int complexityIndex;

    /** Attribute to split on. */
    private int attributeIndex;

    /** Minimum number of objects in a split.   */
    private int minimumNumberOfObjects;

    /** Value of split point. */
    private double splitPoint;

    /** InfoGain of split. */
    private double infoGain;

    /** GainRatio of split.  */
    private double gainRatio;

    /** The sum of the weights of the instances. */
    private double sumOfWeights;

    /** Distribution of class values. */
    protected C45Distribution distribution;

    /**
     * Number of subsets
     */
    protected int numSubsets;

    /** Number of split points. */
    private int index;

    /**
     * Default constructor, used for NoSplit
     */
    public C45SplitModel() {
        attributeIndex = -1;
        minimumNumberOfObjects = 1;
        this.sumOfWeights = 0;
    }

    /**
     * Initializes the split model.
     */
    public C45SplitModel(int attIndex,int minNoObj, double sumOfWeights) {

        // Get index of attribute to split on.
        attributeIndex = attIndex;

        // Set minimum number of objects.
        minimumNumberOfObjects = minNoObj;

        // Set the sum of the weights
        this.sumOfWeights = sumOfWeights;
    }

    /**
     * Returns the number of created subsets for the split.
     * @return int
     */
    public final int getNumberOfSubsets() {
        return numSubsets;
    }

    /**
     * Returns the gain ratio of the current model
     * @return double
     */
    public final double getGainRatio() {
        return gainRatio;
    }

    public final double getInfoGain() {
        return infoGain;
    }

    public final C45Distribution getDistribution() {
        return distribution;
    }

    public final boolean checkModel() {
        return numSubsets > 0;
    }

    public final int getAttributeIndex() {
        return attributeIndex;
    }

    /**
     * Creates a C4.5-type split on the given data. Assumes that none of
     * the class values is missing.
     *
     * @exception Exception if something goes wrong
     */
    public void buildClassifier(Instances trainInstances) throws Exception {
        // Initialize the remaining instance variables.
        numSubsets = 0;
        splitPoint = Double.MAX_VALUE;
        infoGain = 0;
        gainRatio = 0;

        // Different treatment for enumerated and numeric
        // attributes.
        if (trainInstances.attribute(attributeIndex).isNominal()) {
            complexityIndex = trainInstances.attribute(attributeIndex).numValues();
            index = complexityIndex;
            handleEnumeratedAttribute(trainInstances);
        }else{
            complexityIndex = 2;
            index = 0;
            trainInstances.sort(trainInstances.attribute(attributeIndex));
            handleNumericAttribute(trainInstances);
        }
    }

    private void handleEnumeratedAttribute(Instances trainInstances) throws Exception {
        distribution = new C45Distribution(complexityIndex, trainInstances.numClasses());

        // Only Instances with known values are relevant.
        Enumeration enu = trainInstances.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance instance = (Instance) enu.nextElement();
            if (!instance.isMissing(attributeIndex)) {
                distribution.add((int)instance.value(attributeIndex),instance);
            }
        }

        // Check if minimum number of Instances in at least two
        // subsets.
        if (distribution.check(minimumNumberOfObjects)) {
            numSubsets = complexityIndex;
            infoGain = infoGainSplitCritValue(distribution, sumOfWeights);
            gainRatio = gainRatioSplitCritValue(distribution, sumOfWeights, infoGain);
        }
    }

    /**
     * Creates split on numeric attribute.
     *
     * @exception Exception if something goes wrong
     */
    private void handleNumericAttribute(Instances trainInstances) throws Exception {
        int next = 1;
        int last = 0;
        int splitIndex = -1;

        // Current attribute is a numeric attribute.
        distribution = new C45Distribution(2, trainInstances.numClasses());

        // Only Instances with known values are relevant.
        // if the instance has missing value, then we're going to stop there
        // and mark it
        Enumeration enu = trainInstances.enumerateInstances();

        int firstMiss = 0;
        while (enu.hasMoreElements()) {
            Instance instance = (Instance) enu.nextElement();
            if (instance.isMissing(attributeIndex))
                break;
            distribution.add(1, instance);
            firstMiss++;
        }

        // Compute minimum number of Instances required in each
        // subset.
        double minSplit = 0.1 * (distribution.total()) / ((double)trainInstances.numClasses());
        if (Utils.smOrEq(minSplit, minimumNumberOfObjects)) {
            minSplit = minimumNumberOfObjects;
        } else if (Utils.gr(minSplit, 25)) {
            // need to limit the minimum number of instances per subset
            minSplit = 25;
        }

        // Enough Instances with known values?
        if (Utils.sm((double) firstMiss, 2 * minSplit)) {
            return;
        }

        // Compute values of criteria for all possible split
        // indices.
        // use 1e-5 as minimum value
        double defaultEntropy = oldEntropy(distribution);
        while (next < firstMiss) {
            if (trainInstances.instance(next - 1).value(attributeIndex) + 1e-5 <
                    trainInstances.instance(next).value(attributeIndex)) {

                // Move class values for all Instances up to next
                // possible split point.
                distribution.shiftRange(1, 0, trainInstances, last, next);

                // Check if enough Instances in each subset and compute
                // values for criteria.
                if (Utils.grOrEq(distribution.perBag(0), minSplit) && Utils.grOrEq(distribution.perBag(1), minSplit)) {
                    double currentInfoGain = infoGainSplitCritValue(distribution, sumOfWeights, defaultEntropy);
                    if (Utils.gr(currentInfoGain, infoGain)) {
                        infoGain = currentInfoGain;
                        splitIndex = next - 1;
                    }
                    index++;
                }
                last = next;
            }
            next++;
        }

        // Was there any useful split?
        if (index == 0)
            return;

        // Compute modified information gain for best split.
        infoGain -= (Utils.log2(index) / sumOfWeights);
        if (Utils.smOrEq(infoGain, 0))
            return;

        // Set instance variables' values to values for
        // best split.
        numSubsets = 2;
        splitPoint = (trainInstances.instance(splitIndex + 1).value(attributeIndex) +
                trainInstances.instance(splitIndex).value(attributeIndex)) / 2;

        // In case we have a numerical precision problem we need to choose the
        // smaller value
        if (splitPoint == trainInstances.instance(splitIndex + 1).value(attributeIndex)) {
            splitPoint = trainInstances.instance(splitIndex).value(attributeIndex);
        }

        // Restore distribution for best split.
        distribution = new C45Distribution(2,trainInstances.numClasses());
        distribution.addRange(0, trainInstances, 0, splitIndex + 1);
        distribution.addRange(1, trainInstances, splitIndex + 1, firstMiss);

        // Compute modified gain ratio for best split.
        gainRatio = gainRatioSplitCritValue(distribution, sumOfWeights, infoGain);
    }

    /**
     * This method computes the information gain in the same way
     * C4.5 does.
     *
     * @param bags the distribution
     * @param totalNoInst weight of ALL instances (including the
     * ones with missing values).
     */
    public final double infoGainSplitCritValue(C45Distribution bags, double totalNoInst) {
        double noUnknown = totalNoInst-bags.total();
        double unknownRate = noUnknown / totalNoInst;
        double numerator = (oldEntropy(bags) - newEntropy(bags));
        numerator = (1-unknownRate)*numerator;

        // Splits with no gain are useless.
        if (Utils.eq(numerator,0))
            return 0;

        return numerator/bags.total();
    }

    /**
     * This method is a straightforward implementation of the gain
     * ratio criterion for the given distribution.
     */
    public final double gainRatioSplitCritValue(C45Distribution bags) {
        double numerator = oldEntropy(bags) - newEntropy(bags);

        // Splits with no gain are useless.
        if (Utils.eq(numerator, 0))
            return Double.MAX_VALUE;
        double denominator = splitEntropy(bags);

        // Test if split is trivial.
        if (Utils.eq(denominator, 0))
            return Double.MAX_VALUE;

        //  We take the reciprocal value because we want to minimize the
        // splitting criterion's value.
        return denominator / numerator;
    }

    /**
     * This method computes the gain ratio in the same way C4.5 does.
     *
     * @param bags the distribution
     * @param totalnoInst the weight of ALL instances
     * @param numerator the info gain
     */
    public final double gainRatioSplitCritValue(C45Distribution bags, double totalnoInst, double numerator) {
        // Compute split info.
        double denominator = splitEntropy(bags);

        // Test if split is trivial.
        if (Utils.eq(denominator,0))
            return 0;
        denominator /= totalnoInst;

        return numerator / denominator;
    }

    /**
     * This method computes the information gain in the same way
     * C4.5 does.
     *
     * @param bags the distribution
     * @param totalNoInst weight of ALL instances
     * @param oldEnt entropy with respect to "no-split"-model.
     */
    public final double infoGainSplitCritValue(C45Distribution bags,double totalNoInst, double oldEnt) {
        double noUnknown = totalNoInst-bags.total();
        double unknownRate = noUnknown / totalNoInst;
        double numerator = (oldEnt - newEntropy(bags));
        numerator = (1-unknownRate) * numerator;

        // Splits with no gain are useless.
        if (Utils.eq(numerator,0))
            return 0;

        return numerator / bags.total();
    }

    // constant for log of 2
    private static final double log2 = Math.log(2);

    /**
     * Helper method for computing entropy.
     */
    public final double logFunc(double num) {
        // Constant hard coded for efficiency reasons
        if (num < 1e-6)
            return 0;
        else
            return num * Math.log(num)/log2;
    }

    /**
     * Computes entropy of distribution before splitting.
     */
    public final double oldEntropy(C45Distribution bags) {
        double returnValue = 0;

        for (int i = 0; i < bags.numClasses(); i++) {
            returnValue += logFunc(bags.perClass(i));
        }

        return logFunc(bags.total())-returnValue;
    }

    /**
     * Computes entropy of distribution after splitting.
     */
    public final double newEntropy(C45Distribution bags) {

        double returnValue = 0;

        for (int i = 0; i < bags.numBags(); i++){
            for (int j = 0; j < bags.numClasses(); j++) {
                returnValue += logFunc(bags.perClassPerBag(i,j));
            }
            returnValue -= logFunc(bags.perBag(i));
        }
        return -returnValue;
    }

    /**
     * Computes entropy after splitting without considering the
     * class values.
     */
    public final double splitEntropy(C45Distribution bags) {

        double returnValue = 0;

        for (int i = 0; i < bags.numBags(); i++) {
            returnValue += logFunc(bags.perBag(i));
        }

        return logFunc(bags.total())-returnValue;
    }

    /**
     * Returns index of subset instance is assigned to.
     * Returns -1 if instance is assigned to more than one subset.
     *
     * @exception Exception if something goes wrong
     */
    public final int getSubsetIndex(Instance instance) throws Exception {
        if (instance.isMissing(attributeIndex)) {
            return -1;
        } else {
            if (instance.attribute(attributeIndex).isNominal()) {
                return (int)instance.value(attributeIndex);
            } else {
                if (Utils.smOrEq(instance.value(attributeIndex), splitPoint)) {
                    return 0;
                } else {
                    return 1;
                }
            }
        }
    }

    /**
     * Returns weights if instance is assigned to more than one subset.
     * Returns null if instance is only assigned to one subset.
     */
    public double [] weights(Instance instance) {
        double [] weights = null;

        if (instance.isMissing(attributeIndex)) {
            weights = new double [numSubsets];
            for (int i = 0; i < numSubsets; i++) {
                weights[i] = distribution.perBag(i) / distribution.total();
            }
        }

        return weights;
    }

    /**
     * Sets split point to greatest value in given data smaller or equal to
     * old split point
     *
     * @param allInstances
     */
    public final void setSplitPoint(Instances allInstances) {

        double newSplitPoint = -Double.MAX_VALUE;
        double tempValue;
        Instance instance;

        if ((allInstances.attribute(attributeIndex).isNumeric()) &&
                (numSubsets > 1)) {
            Enumeration enu = allInstances.enumerateInstances();
            while (enu.hasMoreElements()) {
                instance = (Instance) enu.nextElement();
                if (!instance.isMissing(attributeIndex)) {
                    tempValue = instance.value(attributeIndex);
                    if (Utils.gr(tempValue, newSplitPoint) && Utils.smOrEq(tempValue,splitPoint)) {
                        newSplitPoint = tempValue;
                    }
                }
            }
            splitPoint = newSplitPoint;
        }
    }

    /**
     * Help method for computing the split entropy.
     */
    private double splitEntropy(C45Distribution bags, double totalNumberOfInstances) {
        double returnValue = 0;
        double noUnknown;

        noUnknown = totalNumberOfInstances - bags.total();
        if (Utils.gr(bags.total(), 0)){
            for (int i = 0; i < bags.numBags(); i++) {
                returnValue -= logFunc(bags.perBag(i));
            }

            returnValue -= logFunc(noUnknown);
            returnValue += logFunc(totalNumberOfInstances);
        }
        return returnValue;
    }

    /**
     * Sets distribution associated with model.
     */
    public void resetDistribution(Instances data) throws Exception {

        Instances insts = new Instances(data, data.numInstances());
        for (int i = 0; i < data.numInstances(); i++) {
            if (getSubsetIndex(data.instance(i)) > -1) {
                insts.add(data.instance(i));
            }
        }
        C45Distribution newDistribution = new C45Distribution(insts, this);
        newDistribution.addInstWithUnknown(data, attributeIndex);

        distribution = newDistribution;
    }

    /**
     * Splits the given set of instances into subsets.
     *
     * @exception Exception if something goes wrong
     */
    public final Instances [] split(Instances data) throws Exception {
        Instances [] instances = new Instances [numSubsets];

        for (int j = 0; j < numSubsets; j++) {
            instances[j] = new Instances(data, data.numInstances());
        }

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = ((Instances) data).instance(i);
            double [] weights = weights(instance);
            int subset = getSubsetIndex(instance);

            if (subset > -1) {
                instances[subset].add(instance);
            } else {
                for (int j = 0; j < numSubsets; j++)
                    if (Utils.gr(weights[j], 0)) {
                        double newWeight = weights[j] * instance.weight();

                        instances[j].add(instance);
                        instances[j].lastInstance().setWeight(newWeight);
                    }
            }
        }

        for (int j = 0; j < numSubsets; j++) {
            instances[j].compactify();
        }

        return instances;
    }

    /**
     * Gets class probability for instance.
     *
     * @exception Exception if something goes wrong
     */
    public final double classProbability(int classIndex, Instance instance, int theSubset) throws Exception {

        if (theSubset <= -1) {
            double [] weights = weights(instance);
            if (weights == null) {
                return distribution.prob(classIndex);
            } else {
                double prob = 0;
                for (int i = 0; i < weights.length; i++) {
                    prob += weights[i] * distribution.prob(classIndex, i);
                }
                return prob;
            }
        } else {
            if (Utils.gr(distribution.perBag(theSubset), 0)) {
                return distribution.prob(classIndex, theSubset);
            } else {
                return distribution.prob(classIndex);
            }
        }
    }

    /**
     * Gets class probability for instance.
     *
     * @exception Exception if something goes wrong
     */
    public double classLaplaceProbability(int classIndex, Instance instance, int theSubset) throws Exception {
        if (theSubset > -1) {
            return distribution.laplaceProb(classIndex, theSubset);
        } else {
            double [] weights = weights(instance);
            if (weights == null) {
                return distribution.laplaceProb(classIndex);
            } else {
                double prob = 0;
                for (int i = 0; i < weights.length; i++) {
                    prob += weights[i] * distribution.laplaceProb(classIndex, i);
                }
                return prob;
            }
        }
    }
}
