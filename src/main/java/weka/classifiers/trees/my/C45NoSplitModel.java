package weka.classifiers.trees.my;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by Alvin Natawiguna on 9/30/2015.
 */
public class C45NoSplitModel extends C45SplitModel {
    /**
     * Creates "no-split"-split for given distribution.
     */
    public C45NoSplitModel(C45Distribution distribution){
        this.distribution = new C45Distribution(distribution);
        numSubsets = 1;
    }

    /**
     * Creates a "no-split"-split for a given set of instances.
     *
     * @exception Exception if split can't be built successfully
     */
    public final void buildClassifier(Instances instances)
            throws Exception {

        distribution = new C45Distribution(instances);
        numSubsets = 1;
    }

    /**
     * Always returns 0 because only there is only one subset.
     */
    public final int whichSubset(Instance instance){

        return 0;
    }

    /**
     * Always returns null because there is only one subset.
     */
    public final double [] weights(Instance instance){

        return null;
    }
}
