package org.docclassification;
import java.util.ArrayList;
import java.util.List;

public class LDAAlgorithm {

    // Define data structures for storing input data
    private List<double[]> positiveSamples; // List of positive class samples
    private List<double[]> negativeSamples; // List of negative class samples

    // Constructor to initialize data structures
    public LDAAlgorithm() {
        positiveSamples = new ArrayList<double[]>();
        negativeSamples = new ArrayList<double[]>();
    }

    // Method to add samples to positive class
    public void addPositiveSample(double[] sample) {
        positiveSamples.add(sample);
    }

    // Method to add samples to negative class
    public void addNegativeSample(double[] sample) {
        negativeSamples.add(sample);
    }

    // Method to calculate LDA scores for binary term feature ranking
    public double[] calculateLDAScores() {
        // Assuming binary features, calculate class means
        double[] positiveMean = calculateMean(positiveSamples);
        double[] negativeMean = calculateMean(negativeSamples);

        // Calculate within-class covariance matrix
        double[] withinClassCovariance = calculateWithinClassCovariance(positiveSamples, positiveMean, negativeSamples, negativeMean);

        // Calculate LDA scores (effectively the difference of means normalized by within-class variance)
        double[] ldaScores = new double[positiveMean.length];
        for (int i = 0; i < positiveMean.length; i++) {
            ldaScores[i] = (positiveMean[i] - negativeMean[i]) / withinClassCovariance[i];
        }

        return ldaScores;
    }

    // Helper method to calculate mean vector
    private double[] calculateMean(List<double[]> samples) {
        int numFeatures = samples.get(0).length;
        double[] mean = new double[numFeatures];

        for (double[] sample : samples) {
            for (int i = 0; i < numFeatures; i++) {
                mean[i] += sample[i];
            }
        }

        for (int i = 0; i < numFeatures; i++) {
            mean[i] /= samples.size();
        }

        return mean;
    }

    // Helper method to calculate within-class covariance matrix
    private double[] calculateWithinClassCovariance(List<double[]> positiveSamples, double[] positiveMean,
                                                    List<double[]> negativeSamples, double[] negativeMean) {
        int numFeatures = positiveMean.length;
        double[] covariance = new double[numFeatures];

        // Calculate covariance for each feature
        for (int i = 0; i < numFeatures; i++) {
            double cov = 0.0;

            // Positive class covariance
            for (double[] sample : positiveSamples) {
                cov += Math.pow(sample[i] - positiveMean[i], 2);
            }

            // Negative class covariance
            for (double[] sample : negativeSamples) {
                cov += Math.pow(sample[i] - negativeMean[i], 2);
            }

            covariance[i] = cov / (positiveSamples.size() + negativeSamples.size() - 2);
        }

        return covariance;
    }

    // Example usage
    public void CallLDA() {
        LDAAlgorithm lda = new LDAAlgorithm();

        // Example data: binary feature vectors
        double[] sample1 = {1, 0, 1, 0, 1, 2}; // Example positive sample
        double[] sample2 = {0, 1, 0, 0, 1, 0}; // Example negative sample

        lda.addPositiveSample(sample1);
        lda.addNegativeSample(sample2);

        // Calculate LDA scores for feature ranking
        double[] ldaScores = lda.calculateLDAScores();

        // Output LDA scores
        System.out.println("LDA Scores:");
        for (int i = 0; i < ldaScores.length; i++) {
            System.out.println("Feature :::: " + i + ": " + ldaScores[i]);
        }
    }
}
