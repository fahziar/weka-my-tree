package weka.classifiers.trees;

import weka.classifiers.Classifier;
import weka.core.*;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * Created by fahziar on 28/09/2015.
 */
public class MyId3 extends Classifier implements Serializable {
    private final double CLASS_VALUE_NOT_LEAF = -99.0;
    private Attribute m_Attribute; //Atrribute of current node
    private MyId3[] m_Branches; //Branches from current node; null if leaf
//    private double[] m_Distribution;
    private double m_ClassValue;
    private Attribute m_ClassAtribute;

    public MyId3(){
        super();
        m_ClassValue = CLASS_VALUE_NOT_LEAF;
    }


    @Override
    public void buildClassifier(Instances instance) throws Exception {
        Instances data = new Instances(instance);
        getCapabilities().testWithFail(data);

        data.deleteWithMissingClass();
        makeTree(data);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities out =  super.getCapabilities();
        out.disableAll();

        out.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        out.enable(Capabilities.Capability.NOMINAL_CLASS);
        out.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return out;
    }

    private double calculateEntropy(Instances data) {
        double classCounts[] = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()){
            Instance i = (Instance) instEnum.nextElement();
            classCounts[(int) i.classValue()]++;
        }

        double entropy = 0.0;
        for (int i=0; i<classCounts.length; i++){
            if (classCounts[i] > 0) {
                double probability = classCounts[i] / data.numInstances();
                entropy -=  probability * Utils.log2(probability);
            }
        }
        return entropy;
    }

    public Instances[] splitInstances(Instances instances, Attribute attribute){
        Instances[] out = new Instances[attribute.numValues()];

        for (int i=0; i<attribute.numValues(); i++){
            out[i] = new Instances(instances, instances.numInstances());
        }

        Enumeration instEnum = instances.enumerateInstances();
        while (instEnum.hasMoreElements()){
            Instance i = (Instance) instEnum.nextElement();
            out[(int) i.value(attribute)].add(i);
        }

        for (int i=0; i<out.length; i++){
            out[i].compactify();
        }

        return out;

    }

    private Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitData[(int) inst.value(att)].add(inst);
        }
        for (int i = 0; i < splitData.length; i++) {
            splitData[i].compactify();
        }
        return splitData;
    }

    private double getInfoGain(double entropy, Instances data, Attribute att){
        double infoGain = entropy;

        Instances[] splitData = splitData(data, att);
        for (int i=0; i<att.numValues(); i++){
            if (splitData[i].numInstances() > 0) {
                try {
                    infoGain -= ((double) splitData[i].numInstances() / data.numInstances()) * calculateEntropy(splitData[i]);
                    if (Double.isNaN(infoGain)){
                        System.out.println("Break");
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        return infoGain;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Id3: no missing values, "
                    + "please.");
        }
        if (isLef()) {
            return m_ClassValue;
        } else {
            return m_Branches[(int) instance.value(m_Attribute)].classifyInstance(instance);
        }
    }

    private void makeTree(Instances data){
        if (data.numInstances() == 0) {
            m_Attribute = null;
            m_ClassValue = Instance.missingValue();
            return;
        }
        double entropy = calculateEntropy(data);
        double infoGain[] = new double[data.numAttributes()];

        //Calculate information gain
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()){
            Attribute att = (Attribute) attEnum.nextElement();
            Instances[] splitData = splitData(data, att);
//            infoGain[att.index()] = getInfoGain(entropy, data.numInstances(), splitData, att);
            try {
//                infoGain[att.index()] = computeInfoGain(data, att);
                infoGain[att.index()] = getInfoGain(entropy, data, att);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        //Select max attribute
        m_Attribute = data.attribute(Utils.maxIndex(infoGain));

        //Make branch or leaf
        if (Utils.eq(infoGain[m_Attribute.index()], 0)){
            m_Attribute = null;

            //create a leaf
            double[] m_Distribution = new double[data.numClasses()];

            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()){
                Instance i = (Instance) instEnum.nextElement();
                m_Distribution[(int) i.classValue()]++;
            }

            Utils.normalize(m_Distribution);
//            m_ClassValue = data.instance(0).classValue();
            m_ClassValue = Utils.maxIndex(m_Distribution);
            m_ClassAtribute = data.classAttribute();

        } else {
            Instances[] split = splitInstances(data, m_Attribute);
            m_Branches = new MyId3[m_Attribute.numValues()];
            for (int i=0; i<m_Attribute.numValues(); i++){
                m_Branches[i] = new MyId3();
                m_Branches[i].makeTree(split[i]);
            }

        }
    }

    @Override
    public String toString() {
        if ((Utils.eq(m_ClassValue, CLASS_VALUE_NOT_LEAF)) && (m_Branches == null)) {
            return "Id3: No model built yet.";
        }
        return "Id3\n\n" + toString(0);
    }

    private String toString(int level) {

        StringBuffer text = new StringBuffer();

        if (m_Attribute == null) {
            text.append(": " + m_ClassAtribute.value((int) m_ClassValue));

        } else {
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
                text.append(m_Branches[j].toString(level + 1));
            }
        }
        return text.toString();
    }

    private boolean isLef(){
        return m_Branches == null;
    }

}
