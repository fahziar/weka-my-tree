/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Distribution.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.my;

import weka.core.*;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * Class for handling a distribution of class values.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Alvin Natawiguna
 */
public class C45Distribution implements Cloneable, Serializable, RevisionHandler {

    /**
     * for serialization
     */
    private static final long serialVersionUID = 8526859638230801376L;

    /**
     * Weight of instances per class per bag.
     */
    private double instancePerClassPerBag[][];

    /**
     * Weight of instances per bag.
     */
    private double weightPerBag[];

    /**
     * Weight of instances per class.
     */
    private double weightPerClass[];

    /**
     * Total weight of instances.
     */
    private double totalWeight;

    /**
     * Creates and initializes a new distribution.
     */
    public C45Distribution(int numBags, int numClasses) {
        instancePerClassPerBag = new double[numBags][0];
        weightPerBag = new double[numBags];
        weightPerClass = new double[numClasses];
        for (int i = 0; i < numBags; i++) {
            instancePerClassPerBag[i] = new double[numClasses];
        }
        totalWeight = 0;
    }

    /**
     * Creates and initializes a new distribution using the given
     * array. WARNING: it just copies a reference to this array.
     */
    public C45Distribution(double[][] table) {
        instancePerClassPerBag = table;
        weightPerBag = new double[table.length];
        weightPerClass = new double[table[0].length];
        for (int i = 0; i < table.length; i++) {
            for (int j = 0; j < table[i].length; j++) {
                weightPerBag[i] += table[i][j];
                weightPerClass[j] += table[i][j];
                totalWeight += table[i][j];
            }
        }
    }

    /**
     * Creates a distribution with only one bag according
     * to instances in source.
     *
     * @throws Exception if something goes wrong
     */
    public C45Distribution(Instances source) throws Exception {
        instancePerClassPerBag = new double[1][0];
        weightPerBag = new double[1];
        totalWeight = 0;
        weightPerClass = new double[source.numClasses()];
        instancePerClassPerBag[0] = new double[source.numClasses()];

        Enumeration enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            add(0, (Instance) enu.nextElement());
        }
    }

    /**
     * Creates a distribution according to given instances and
     * split model.
     *
     * @throws Exception if something goes wrong
     */
    public C45Distribution(Instances source, C45SplitModel modelToUse) throws Exception {
        int countSubsets = modelToUse.getNumberOfSubsets();
        int countClasses = source.numClasses();

        instancePerClassPerBag = new double[countSubsets][0];
        weightPerBag = new double[countSubsets];
        totalWeight = 0;
        weightPerClass = new double[countClasses];

        for (int i = 0; i < countSubsets; i++) {
            instancePerClassPerBag[i] = new double[countClasses];
        }

        Enumeration enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance instance = (Instance) enu.nextElement();
            int index = modelToUse.getSubsetIndex(instance);
            if (index != -1)
                add(index, instance);
            else {
                double[] weights = modelToUse.weights(instance);
                addWeights(instance, weights);
            }
        }
    }

    /**
     * Creates distribution with only one bag by merging all
     * bags of given distribution.
     */
    public C45Distribution(C45Distribution toMerge) {
        totalWeight = toMerge.totalWeight;
        weightPerClass = new double[toMerge.numClasses()];
        System.arraycopy(toMerge.weightPerClass, 0, weightPerClass, 0, toMerge.numClasses());

        instancePerClassPerBag = new double[1][0];
        instancePerClassPerBag[0] = new double[toMerge.numClasses()];
        System.arraycopy(toMerge.weightPerClass, 0, instancePerClassPerBag[0], 0, toMerge.numClasses());

        weightPerBag = new double[1];
        weightPerBag[0] = totalWeight;
    }

    /**
     * Creates distribution with two bags by merging all bags apart of
     * the indicated one.
     */
    public C45Distribution(C45Distribution toMerge, int index) {
        totalWeight = toMerge.totalWeight;
        weightPerClass = new double[toMerge.numClasses()];
        System.arraycopy(toMerge.weightPerClass, 0, weightPerClass, 0, toMerge.numClasses());

        instancePerClassPerBag = new double[2][0];
        instancePerClassPerBag[0] = new double[toMerge.numClasses()];
        System.arraycopy(toMerge.instancePerClassPerBag[index], 0, instancePerClassPerBag[0], 0, toMerge.numClasses());

        instancePerClassPerBag[1] = new double[toMerge.numClasses()];
        for (int i = 0; i < toMerge.numClasses(); i++) {
            instancePerClassPerBag[1][i] = toMerge.weightPerClass[i] - instancePerClassPerBag[0][i];
        }

        weightPerBag = new double[2];
        weightPerBag[0] = toMerge.weightPerBag[index];
        weightPerBag[1] = totalWeight - weightPerBag[0];
    }

    /**
     * Returns number of non-empty bags of distribution.
     */
    public final int actualNumBags() {
        int returnValue = 0;

        for (double weight : weightPerBag) {
            if (Utils.gr(weight, 0)) {
                returnValue++;
            }
        }

        return returnValue;
    }

    /**
     * Returns number of classes actually occuring in distribution.
     */
    public final int actualNumClasses() {
        int returnValue = 0;

        for (double weight : weightPerClass) {
            if (Utils.gr(weight, 0)) {
                returnValue++;
            }
        }

        return returnValue;
    }

    /**
     * Returns number of classes actually occuring in given bag.
     */
    public final int actualNumClasses(int bagIndex) {
        int returnValue = 0;
        for (int i = 0; i < weightPerClass.length; i++) {
            if (Utils.gr(instancePerClassPerBag[bagIndex][i], 0)) {
                returnValue++;
            }
        }

        return returnValue;
    }

    /**
     * Adds given instance to given bag.
     *
     * @throws Exception if something goes wrong
     */
    public final void add(int bagIndex, Instance instance) throws Exception {
        int classIndex = (int) instance.classValue();
        double weight = instance.weight();

        instancePerClassPerBag[bagIndex][classIndex] += weight;
        weightPerBag[bagIndex] += weight;
        weightPerClass[classIndex] += weight;
        totalWeight += weight;
    }

    /**
     * Subtracts given instance from given bag.
     *
     * @throws Exception if something goes wrong
     */
    public final void sub(int bagIndex, Instance instance) throws Exception {
        int classIndex = (int) instance.classValue();
        double weight = instance.weight();

        instancePerClassPerBag[bagIndex][classIndex] -= weight;
        weightPerBag[bagIndex] -= weight;
        weightPerClass[classIndex] -= weight;
        totalWeight -= weight;
    }

    /**
     * Adds counts to given bag.
     */
    public final void add(int bagIndex, double[] counts) {
        double sum = Utils.sum(counts);

        for (int i = 0; i < counts.length; i++) {
            instancePerClassPerBag[bagIndex][i] += counts[i];
        }
        weightPerBag[bagIndex] += sum;

        for (int i = 0; i < counts.length; i++) {
            weightPerClass[i] += counts[i];
        }

        totalWeight += sum;
    }

    /**
     * Adds all instances with unknown values for given attribute, weighted
     * according to frequency of instances in each bag.
     *
     * @throws Exception if something goes wrong
     */
    public final void addInstWithUnknown(Instances source, int attIndex) throws Exception {

        double[] probs = new double[weightPerBag.length];
        for (int j = 0; j < weightPerBag.length; j++) {
            if (Utils.eq(totalWeight, 0)) {
                probs[j] = 1.0 / probs.length;
            } else {
                probs[j] = weightPerBag[j] / totalWeight;
            }
        }

        Enumeration enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance instance = (Instance) enu.nextElement();

            if (instance.isMissing(attIndex)) {
                // handle missing value
                int classIndex = (int) instance.classValue();

                double weight = instance.weight();
                weightPerClass[classIndex] += weight;
                totalWeight += weight;
                for (int i = 0; i < weightPerBag.length; i++) {
                    double newWeight = probs[i] * weight;
                    instancePerClassPerBag[i][classIndex] += newWeight;
                    weightPerBag[i] += newWeight;
                }
            }
        }
    }

    /**
     * Adds all instances in given range to given bag.
     *
     * @throws Exception if something goes wrong
     */
    public final void addRange(int bagIndex, Instances source, int startIndex, int lastPlusOne) throws Exception {
        double sumOfWeights = 0;

        for (int i = startIndex; i < lastPlusOne; i++) {
            Instance instance = (Instance) source.instance(i);
            int classIndex = (int) instance.classValue();

            sumOfWeights += instance.weight();
            instancePerClassPerBag[bagIndex][classIndex] += instance.weight();
            weightPerClass[classIndex] += instance.weight();
        }

        weightPerBag[bagIndex] += sumOfWeights;
        totalWeight += sumOfWeights;
    }

    /**
     * Adds given instance to all bags weighting it according to given weights.
     *
     * @throws Exception if something goes wrong
     */
    public final void addWeights(Instance instance, double[] weights) throws Exception {
        int classIndex = (int) instance.classValue();

        for (int i = 0; i < weightPerBag.length; i++) {
            double weight = instance.weight() * weights[i];
            instancePerClassPerBag[i][classIndex] += weight;
            weightPerBag[i] += weight;
            weightPerClass[classIndex] += weight;
            totalWeight += weight;
        }
    }

    /**
     * Checks if at least two bags contain a minimum number of instances.
     */
    public final boolean check(double minNoObj) {
        int counter = 0;

        for (double weight : weightPerBag) {
            if (Utils.grOrEq(weight, minNoObj)) {
                counter++;

                // optimization
                if (counter == 2) {
                    break;
                }
            }
        }

        return counter > 1;
    }

    /**
     * Clones distribution (Deep copy of distribution).
     */
    public final Object clone() throws CloneNotSupportedException {
        super.clone();

        C45Distribution newDistribution = new C45Distribution(weightPerBag.length, weightPerClass.length);
        for (int i = 0; i < weightPerBag.length; i++) {
            newDistribution.weightPerBag[i] = weightPerBag[i];
            System.arraycopy(instancePerClassPerBag[i], 0, newDistribution.instancePerClassPerBag[i], 0, weightPerClass.length);
        }

        System.arraycopy(weightPerClass, 0, newDistribution.weightPerClass, 0, weightPerClass.length);

        newDistribution.totalWeight = totalWeight;

        return newDistribution;
    }

    /**
     * Deletes given instance from given bag.
     *
     * @throws Exception if something goes wrong
     */
    public final void del(int bagIndex, Instance instance) throws Exception {
        int classIndex = (int) instance.classValue();
        double weight = instance.weight();

        instancePerClassPerBag[bagIndex][classIndex] -= weight;
        weightPerBag[bagIndex] -= weight;
        weightPerClass[classIndex] -= weight;
        totalWeight -= weight;
    }

    /**
     * Deletes all instances in given range from given bag.
     *
     * @throws Exception if something goes wrong
     */
    public final void delRange(int bagIndex, Instances source, int startIndex, int lastPlusOne) throws Exception {
        double sumOfWeights = 0;

        for (int i = startIndex; i < lastPlusOne; i++) {
            Instance instance = (Instance) source.instance(i);
            int classIndex = (int) instance.classValue();
            sumOfWeights += instance.weight();
            instancePerClassPerBag[bagIndex][classIndex] -= instance.weight();
            weightPerClass[classIndex] -= instance.weight();
        }

        weightPerBag[bagIndex] -= sumOfWeights;
        totalWeight -= sumOfWeights;
    }

    /**
     * Sets all counts to zero.
     */
    public final void initialize() {
        for (int i = 0; i < weightPerClass.length; i++) {
            weightPerClass[i] = 0;
        }

        for (int i = 0; i < weightPerBag.length; i++) {
            weightPerBag[i] = 0;
        }

        for (int i = 0; i < weightPerBag.length; i++) {
            for (int j = 0; j < weightPerClass.length; j++) {
                instancePerClassPerBag[i][j] = 0;
            }
        }

        totalWeight = 0;
    }

    /**
     * Returns matrix with distribution of class values.
     */
    public final double[][] matrix() {
        return instancePerClassPerBag;
    }

    /**
     * Returns index of bag containing maximum number of instances.
     */
    public final int maxBag() {
        double max = 0;
        int maxIndex = -1;

        for (int i = 0; i < weightPerBag.length; i++) {
            if (Utils.grOrEq(weightPerBag[i], max)) {
                max = weightPerBag[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /**
     * Returns class with highest frequency over all bags.
     */
    public final int maxClass() {
        double maxCount = 0;
        int maxIndex = 0;
        int i;

        for (i = 0; i < weightPerClass.length; i++)
            if (Utils.gr(weightPerClass[i], maxCount)) {
                maxCount = weightPerClass[i];
                maxIndex = i;
            }

        return maxIndex;
    }

    /**
     * Returns class with highest frequency for given bag.
     */
    public final int maxClass(int index) {
        if (Utils.gr(weightPerBag[index], 0)) {
            int maxIndex = 0;
            double maxCount = 0;

            for (int i = 0; i < weightPerClass.length; i++) {
                if (Utils.gr(instancePerClassPerBag[index][i], maxCount)) {
                    maxCount = instancePerClassPerBag[index][i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        } else {
            return maxClass();
        }
    }

    /**
     * Returns number of bags.
     */
    public final int numBags() {
        return weightPerBag.length;
    }

    /**
     * Returns number of classes.
     */
    public final int numClasses() {
        return weightPerClass.length;
    }

    /**
     * Returns perClass(maxClass()).
     */
    public final double numCorrect() {
        return weightPerClass[maxClass()];
    }

    /**
     * Returns perClassPerBag(index,maxClass(index)).
     */
    public final double numCorrect(int index) {
        return instancePerClassPerBag[index][maxClass(index)];
    }

    /**
     * Returns totalWeight-numCorrect().
     */
    public final double numIncorrect() {
        return totalWeight - numCorrect();
    }

    /**
     * Returns weightPerBag(index)-numCorrect(index).
     */
    public final double numIncorrect(int index) {
        return weightPerBag[index] - numCorrect(index);
    }

    /**
     * Returns number of (possibly fractional) instances of given class in
     * given bag.
     */
    public final double perClassPerBag(int bagIndex, int classIndex) {
        return instancePerClassPerBag[bagIndex][classIndex];
    }

    /**
     * Returns number of (possibly fractional) instances in given bag.
     */
    public final double perBag(int bagIndex) {
        return weightPerBag[bagIndex];
    }

    /**
     * Returns number of (possibly fractional) instances of given class.
     */
    public final double perClass(int classIndex) {
        return weightPerClass[classIndex];
    }

    /**
     * Returns relative frequency of class over all bags with
     * Laplace correction.
     */
    public final double laplaceProb(int classIndex) {
        return (weightPerClass[classIndex] + 1) / (totalWeight + (double) weightPerClass.length);
    }

    /**
     * Returns relative frequency of class for given bag.
     */
    public final double laplaceProb(int classIndex, int intIndex) {
        if (Utils.gr(weightPerBag[intIndex], 0))
            return (instancePerClassPerBag[intIndex][classIndex] + 1.0) /
                    (weightPerBag[intIndex] + (double) weightPerClass.length);
        else
            return laplaceProb(classIndex);

    }

    /**
     * Returns relative frequency of class over all bags.
     */
    public final double prob(int classIndex) {
        if (Utils.eq(totalWeight, 0)) {
            return 0;
        } else {
            return weightPerClass[classIndex] / totalWeight;
        }
    }

    /**
     * Returns relative frequency of class for given bag.
     */
    public final double prob(int classIndex, int intIndex) {
        if (Utils.gr(weightPerBag[intIndex], 0)) {
            return instancePerClassPerBag[intIndex][classIndex] / weightPerBag[intIndex];
        } else {
            return prob(classIndex);
        }
    }

    /**
     * Subtracts the given distribution from this one. The results
     * has only one bag.
     */
    public final C45Distribution subtract(C45Distribution toSubstract) {
        C45Distribution newDist = new C45Distribution(1, weightPerClass.length);

        newDist.weightPerBag[0] = totalWeight - toSubstract.totalWeight;
        newDist.totalWeight = newDist.weightPerBag[0];

        for (int i = 0; i < weightPerClass.length; i++) {
            newDist.instancePerClassPerBag[0][i] = weightPerClass[i] - toSubstract.weightPerClass[i];
            newDist.weightPerClass[i] = newDist.instancePerClassPerBag[0][i];
        }

        return newDist;
    }

    /**
     * Returns totalWeight number of (possibly fractional) instances.
     */
    public final double total() {
        return totalWeight;
    }

    /**
     * Shifts given instance from one bag to another one.
     *
     * @throws Exception if something goes wrong
     */
    public final void shift(int from, int to, Instance instance) throws Exception {
        int classIndex = (int) instance.classValue();
        double weight = instance.weight();

        instancePerClassPerBag[from][classIndex] -= weight;
        instancePerClassPerBag[to][classIndex] += weight;
        weightPerBag[from] -= weight;
        weightPerBag[to] += weight;
    }

    /**
     * Shifts all instances in given range from one bag to another one.
     *
     * @throws Exception if something goes wrong
     */
    public final void shiftRange(int from, int to, Instances source, int startIndex, int lastPlusOne) throws Exception {
        for (int i = startIndex; i < lastPlusOne; i++) {
            Instance instance = (Instance) source.instance(i);
            int classIndex = (int) instance.classValue();
            double weight = instance.weight();

            instancePerClassPerBag[from][classIndex] -= weight;
            instancePerClassPerBag[to][classIndex] += weight;
            weightPerBag[from] -= weight;
            weightPerBag[to] += weight;
        }
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1.1201 $");
    }
}
