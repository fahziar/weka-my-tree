package weka.classifiers.trees.my;

import weka.classifiers.Classifier;
import weka.core.*;

/**
 * Created by Alvin Natawiguna on 9/28/2015.
 *
 * Represents the J48 tree, including nodes
 */
public class J48 extends Classifier {

    /**
     * Children of the tree
     */
    protected J48 children[];

    /**
     * Ways to select model
     */
    protected C45ModelSelection modelSelection;

    /**
     * Local model at node
     */
    protected C45SplitModel localModel;

    /**
     * Confidence level for pruning
     */
    protected float confidence = 0.25f;

    /** True if the tree is to be pruned. */
    protected boolean prune = false;

    /** Is subtree raising to be performed? */
    protected boolean raiseSubtree = true;

    /** Cleanup after the tree has been built. */
    protected boolean cleanup = true;

    /**
     * True if node is a leaf
     */
    protected boolean leaf;

    /**
     * True if node is empty
     */
    protected boolean empty;

    /**
     * Training instances
     */
    protected Instances trainInstance;

    /**
     * Instances for pruning
     */
    protected C45Distribution test;

    /**
     * id of the node
     */
    protected int id;

    public J48() {
        super();
    }

    public J48(C45ModelSelection model) {
        modelSelection = model;
    }

    public Capabilities getCapabilities() {
        Capabilities capabilities = new Capabilities(this);
        capabilities.disableAll();

        // attributes
        capabilities.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        capabilities.enable(Capabilities.Capability.NOMINAL_CLASS);
        capabilities.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        capabilities.setMinimumNumberInstances(0);

        return capabilities;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier tree handle the data?
        getCapabilities().testWithFail(data);

        // set the initial model
        if (modelSelection == null) {
            modelSelection = new C45ModelSelection(2, data);
        }

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        buildTree(data, raiseSubtree || !cleanup);
        collapse();
        if (prune) {
            prune();
        }
        if (cleanup) {
            cleanup(new Instances(data, 0));
        }
    }

    /**
     * Collapses a tree to a node if training error doesn't increase.
     */
    public final void collapse() {
        if (!leaf){
            double errorsOfSubtree = getTrainingErrors();
            double errorsOfTree = localModel.getDistribution().numIncorrect();
            if (errorsOfSubtree >= errorsOfTree - 1E-3){

                // Free adjacent trees
                children = null;
                leaf = true;

                // Get NoSplit Model for tree.
                localModel = new C45NoSplitModel(localModel.getDistribution());
            } else {
                for (J48 child: children) {
                    child.collapse();
                }
            }
        }
    }

    /**
     * Prunes a tree using C4.5's pruning procedure.
     *
     * @throws Exception if something goes wrong
     */
    public void prune() throws Exception {
        if (!leaf) {

            // Prune all subtrees.
            for (J48 child: children) {
                child.prune();
            }

            // Compute error for largest branch
            int indexOfLargestBranch = localModel.getDistribution().maxBag();
            double errorsLargestBranch;

            if (raiseSubtree) {
                errorsLargestBranch = children[indexOfLargestBranch]
                        .getEstimatedErrorsForBranch(trainInstance);
            } else {
                errorsLargestBranch = Double.MAX_VALUE;
            }

            // Compute error if this Tree would be leaf
            double errorsLeaf = getEstimatedErrorsForDistribution(localModel.getDistribution());

            // Compute error for the whole subtree
            double errorsTree = getEstimatedErrors();

            // Decide if leaf is best choice.
            if (Utils.smOrEq(errorsLeaf, errorsTree + 0.1) &&
                    Utils.smOrEq(errorsLeaf, errorsLargestBranch + 0.1))
            {

                // Free son Trees
                children = null;
                leaf = true;

                // Get NoSplit Model for node.
                localModel = new C45NoSplitModel(localModel.getDistribution());
                return;
            }

            // Decide if largest branch is better choice
            // than whole subtree.
            if (Utils.smOrEq(errorsLargestBranch, errorsTree + 0.1)){
                J48 largestBranch = children[indexOfLargestBranch];

                children = largestBranch.children;
                localModel = largestBranch.localModel;
                leaf = largestBranch.leaf;

                newDistribution(trainInstance);
                prune();
            }
        }
    }

    /**
     * Builds the tree structure.
     *
     * @param data the data for which the tree structure is to be
     * generated.
     * @param keepData is training data to be kept?
     * @throws Exception if something goes wrong
     */
    public void buildTree(Instances data, boolean keepData) throws Exception {
        if (keepData) {
            trainInstance = data;
        }

        // initialize remaining attributes
        test = null;
        leaf = false;
        empty = false;
        children = null;
        localModel = modelSelection.selectModel(data);

        assert localModel != null;
        if (localModel.getNumberOfSubsets() > 1) {
            Instances []localInstances = localModel.split(data);
            data = null;

            children = new J48[localModel.getNumberOfSubsets()];
            for (int i = 0; i < children.length; i++) {
                children[i] = getNewTree(localInstances[i]);
                localInstances[i] = null;
            }
        } else {
            leaf = true;
            if (Utils.eq(data.sumOfWeights(), 0)) {
                empty = true;
            }

            data = null;
        }
    }

    /**
     * Builds the tree structure with hold out set
     *
     * @param train the data for which the tree structure is to be
     * generated.
     * @param test the test data for potential pruning
     * @param keepData is training Data to be kept?
     * @throws Exception if something goes wrong
     */
    public void buildTree(Instances train, Instances test, boolean keepData) throws Exception {
        if (keepData) {
            trainInstance = train;
        }

        leaf = false;
        empty = false;
        children = null;
        this.test = new C45Distribution(test, localModel);

        localModel = modelSelection.selectModel(train, test);
        if (localModel.getNumberOfSubsets() > 1) {
            Instances []localTrain = localModel.split(train);
            Instances []localTest = localModel.split(test);
            train = test = null;

            children = new J48 [localModel.getNumberOfSubsets()];
            for (int i = 0; i < children.length; i++) {
                children[i] = getNewTree(localTrain[i], localTest[i]);
                localTrain[i] = null;
                localTest[i] = null;
            }

        } else {
            leaf = true;
            if (Utils.eq(train.sumOfWeights(), 0)) {
                empty = true;
            }

            train = test = null;
        }
    }

    /**
     * Classifies an instance.
     *
     * @param instance the instance to classify
     * @return the classification
     * @throws Exception if something goes wrong
     */
    public double classifyInstance(Instance instance) throws Exception {

        double maxProb = -1;
        int maxIndex = 0;

        for (int j = 0; j < instance.numClasses(); j++) {
            double currentProb = getInstanceProbability(j, instance, 1);
            if (Utils.gr(currentProb, maxProb)) {
                maxIndex = j;
                maxProb = currentProb;
            }
        }

        return (double)maxIndex;
    }

    /**
     * Cleanup in order to save memory.
     *
     * @param justHeaderInfo
     */
    public final void cleanup(Instances justHeaderInfo) {
        trainInstance = justHeaderInfo;
        test = null;
        if (!leaf) {
            for (J48 child : children) {
                child.cleanup(justHeaderInfo);
            }
        }
    }

    /**
     * Returns class probabilities for a weighted instance.
     *
     * @param instance the instance to get the distribution for
     * @param useLaplace whether to use laplace or not
     * @return the distribution
     * @throws Exception if something goes wrong
     */
    public final double []distributionForInstance(Instance instance, boolean useLaplace)
            throws Exception {

        double [] doubles = new double[instance.numClasses()];

        for (int i = 0; i < doubles.length; i++) {
            if (!useLaplace) {
                doubles[i] = getInstanceProbability(i, instance, 1);
            } else {
                doubles[i] = getInstanceLaplaceProbability(i, instance, 1);
            }
        }

        return doubles;
    }

    /**
     * Returns number of leaves in tree structure.
     *
     * @return the number of leaves
     */
    public int numLeaves() {
        int num = 0;

        if (leaf) {
            num = 1;
        } else {
            for (J48 child: children) {
                num += child.numLeaves();
            }
        }

        return num;
    }

    /**
     * Returns number of nodes in tree structure.
     *
     * @return the number of nodes
     */
    public int numNodes() {
        int no = 1;

        if (!leaf) {
            for (J48 child: children) {
                no += child.numNodes();
            }
        }

        return no;
    }

    /**
     * Returns a newly created tree.
     *
     * @param data the training data
     * @return the generated tree
     * @throws Exception if something goes wrong
     */
    protected J48 getNewTree(Instances data) throws Exception {
        J48 tree = new J48(modelSelection);
        tree.buildTree(data, false);

        return tree;
    }

    /**
     * Returns a newly created tree.
     *
     * @param train the training data
     * @param test the pruning data.
     * @return the generated tree
     * @throws Exception if something goes wrong
     */
    protected J48 getNewTree(Instances train, Instances test) throws Exception {
        J48 newTree = new J48(modelSelection);
        newTree.buildTree(train, test, false);

        return newTree;
    }

    /**
     * Help method for computing class probabilities of
     * a given instance.
     *
     * @param classIndex the class index
     * @param instance the instance to compute the probabilities for
     * @param weight the weight to use
     * @return the laplace probs
     * @throws Exception if something goes wrong
     */
    private double getInstanceLaplaceProbability(int classIndex, Instance instance, double weight)
            throws Exception {

        double prob = 0;

        if (leaf) {
            return weight * localModel.classLaplaceProbability(classIndex, instance, -1);
        } else {
            int treeIndex = localModel.getSubsetIndex(instance);
            if (treeIndex == -1) {
                double[] weights = localModel.weights(instance);
                for (int i = 0; i < children.length; i++) {
                    if (children[i].empty) {
                        prob += children[i].getInstanceLaplaceProbability(classIndex, instance,
                                weights[i] * weight);
                    }
                }
                return prob;
            } else {
                if (children[treeIndex].empty) {
                    return weight * localModel.classLaplaceProbability(classIndex, instance,
                            treeIndex);
                } else {
                    return children[treeIndex].getInstanceLaplaceProbability(classIndex, instance, weight);
                }
            }
        }
    }

    /**
     * Help method for computing class probabilities of
     * a given instance.
     *
     * @param classIndex the class index
     * @param instance the instance to compute the probabilities for
     * @param weight the weight to use
     * @return the probs
     * @throws Exception if something goes wrong
     */
    private double getInstanceProbability(int classIndex, Instance instance, double weight) throws Exception {
        double prob = 0;

        if (leaf) {
            prob = weight * localModel.classProbability(classIndex, instance, -1);
        } else {
            int treeIndex = localModel.getSubsetIndex(instance);
            if (treeIndex == -1) {
                double[] weights = localModel.weights(instance);
                for (int i = 0; i < children.length; i++) {
                    if (children[i].empty) {
                        prob += children[i].getInstanceProbability(classIndex, instance,
                                weights[i] * weight);
                    }
                }
            } else {
                if (children[treeIndex].empty) {
                    prob = weight * localModel.classProbability(classIndex, instance, treeIndex);
                } else {
                    prob = children[treeIndex].getInstanceProbability(classIndex, instance, weight);
                }
            }
        }

        return prob;
    }

    /**
     * Computes estimated errors for tree.
     *
     * @return the estimated errors
     */
    private double getEstimatedErrors(){
        double errors = 0;

        if (leaf) {
            errors = getEstimatedErrorsForDistribution(localModel.getDistribution());
        } else {
            for (J48 child: children) {
                errors += child.getEstimatedErrors();
            }
        }

        return errors;
    }

    /**
     * Computes estimated errors for one branch.
     *
     * @param data the data to work with
     * @return the estimated errors
     * @throws Exception if something goes wrong
     */
    private double getEstimatedErrorsForBranch(Instances data) throws Exception {
        double errors = 0;

        if (leaf) {
            errors = getEstimatedErrorsForDistribution(new C45Distribution(data));
        } else {
            C45Distribution savedDist = localModel.getDistribution();
            localModel.resetDistribution(data);

            Instances []localInstances = localModel.split(data);
            localModel.distribution = savedDist;

            for (int i = 0; i < children.length; i++) {
                errors += children[i].getEstimatedErrorsForBranch(localInstances[i]);
            }
        }

        return errors;
    }

    /**
     * Computes estimated errors for leaf.
     *
     * @param dist the distribution to use
     * @return the estimated errors
     */
    private double getEstimatedErrorsForDistribution(C45Distribution dist) {
        if (Utils.eq(dist.total(), 0)) {
            return 0;
        } else {
            return dist.numIncorrect() + countExtraError(dist.total(), dist.numIncorrect(), confidence);
        }
    }

    /**
     * Computes errors of tree on training data.
     *
     * @return the training errors
     */
    private double getTrainingErrors() {
        double errors = 0;

        if (leaf) {
            errors = localModel.getDistribution().numIncorrect();
        } else {
            for(J48 child: children) {
                errors += child.getTrainingErrors();
            }

        }

        return errors;
    }

    /**
     * Computes new distributions of instances for nodes
     * in tree.
     *
     * @param data the data to compute the distributions for
     * @throws Exception if something goes wrong
     */
    private void newDistribution(Instances data) throws Exception {
        localModel.resetDistribution(data);
        trainInstance = data;

        if (!leaf) {
            Instances []localInstances = localModel.split(data);
            for (int i = 0; i < children.length; i++) {
                children[i].newDistribution(localInstances[i]);
            }
        } else {
            // Check whether there are some instances at the leaf now!
            if (!Utils.eq(data.sumOfWeights(), 0)) {
                empty = false;
            }
        }
    }

    /**
     * Computes estimated extra error for given total number of instances
     * and error using normal approximation to binomial distribution
     * (and continuity correction).
     *
     * @see weka.classifiers.trees.j48.Stats
     * @param instanceCount number of instances
     * @param observedError observed error
     * @param confidence confidence value
     * @return extra error value
     */
    private double countExtraError(double instanceCount, double observedError, double confidence) {

        // you must be joking.. the confidence is over 9000!
        if (confidence > 0.5) {
            System.err.println("WARNING: confidence value for pruning " +
                    " too high. Error estimate not modified.");
            return 0;
        }

        // Check for extreme cases at the low end because the
        // normal approximation won't work
        if (observedError < 1) {

            // Base case (i.e. e == 0) from document Geigy Scientific
            // Tables, 6th edition, page 185
            double base = instanceCount * (1 - Math.pow(confidence, 1 / instanceCount));
            if (observedError == 0) {
                return base;
            }

            return base + observedError * (countExtraError(instanceCount, 1, confidence) - base);
        }

        // Use linear interpolation at the high end (i.e. between N - 0.5
        // and N) because of the continuity correction
        if (observedError + 0.5 >= instanceCount) {

            // Make sure that we never return anything smaller than zero
            return Math.max(instanceCount - observedError, 0);
        }

        // Get z-score corresponding to CF
        double z = Statistics.normalInverse(1 - confidence);

        // Compute upper limit of confidence interval
        double  f = (observedError + 0.5) / instanceCount;
        double r = (f + (z * z) / (2 * observedError) +
                z * Math.sqrt((f / instanceCount) -
                        (f * f / instanceCount) +
                        (z * z / (4 * instanceCount * instanceCount)))) /
                (1 + (z * z) / instanceCount);

        return (r * instanceCount) - observedError;
    }
}
