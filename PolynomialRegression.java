class PolynomialRegression {
    private final int degree;
    private double[] coefficients;

    // Constructor que inicializa el grado del polinomio y crea un arreglo para los coeficientes
    public PolynomialRegression(int degree) {
        this.degree = degree;
        this.coefficients = new double[degree + 1];
    }

    public void fit(double[] x, double[] y) {
        int n = x.length;
        double[][] X = new double[n][degree + 1];
        double[] Y = y;

        // Construir la matriz
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= degree; j++) {
                X[i][j] = Math.pow(x[i], j);
            }
        }

        double[][] XT = transpose(X);
        double[][] XTX = multiply(XT, X);
        double[] XTY = multiplyVectorMatrix(XT, Y);

        coefficients = gaussElimination(XTX, XTY);
    }

    public double predict(double x) {
        double y = 0;
        for (int i = 0; i <= degree; i++) {
            y += coefficients[i] * Math.pow(x, i);
        }
        return y;
    }

    public double r2(double[] x, double[] y) {
        double ssTotal = 0;
        double ssResidual = 0;
        double meanY = calculateMean(y);

        for (int i = 0; i < x.length; i++) {
            double predictedY = predict(x[i]);
            ssTotal += Math.pow(y[i] - meanY, 2);
            ssResidual += Math.pow(y[i] - predictedY, 2);
        }

        return 1 - (ssResidual / ssTotal);
    }

    public double[] getCoefficients() {
        return coefficients;
    }

    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposedMatrix = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposedMatrix[j][i] = matrix[i][j];
            }
        }

        return transposedMatrix;
    }

    private double[] multiplyVectorMatrix(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] result = new double[rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }

        return result;
    }

    private double[][] multiply(double[][] matrix1, double[][] matrix2) {
        int rows1 = matrix1.length;
        int cols1 = matrix1[0].length;
        int cols2 = matrix2[0].length;
        double[][] result = new double[rows1][cols2];

        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < cols2; j++) {
                for (int k = 0; k < cols1; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }

        return result;
    }

    private double[] gaussElimination(double[][] A, double[] b) {
        int n = b.length;
        for (int i = 0; i < n; i++) {
            // Buscar el máximo en esta columna
            double maxEl = Math.abs(A[i][i]);
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(A[k][i]) > maxEl) {
                    maxEl = Math.abs(A[k][i]);
                    maxRow = k;
                }
            }

            // Intercambiar la fila máxima con la fila actual (columna por columna)
            double[] temp = A[maxRow];
            A[maxRow] = A[i];
            A[i] = temp;
            double t = b[maxRow];
            b[maxRow] = b[i];
            b[i] = t;

            // Hacer todas las filas por debajo de esta 0 en la columna actual
            for (int k = i + 1; k < n; k++) {
                double c = -A[k][i] / A[i][i];
                for (int j = i; j < n; j++) {
                    if (i == j) {
                        A[k][j] = 0;
                    } else {
                        A[k][j] += c * A[i][j];
                    }
                }
                b[k] += c * b[i];
            }
        }

        // Resolver la ecuación Ax=b para una matriz triangular superior A
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = b[i] / A[i][i];
            for (int k = i - 1; k >= 0; k--) {
                b[k] -= A[k][i] * x[i];
            }
        }
        return x;
    }

    private double calculateMean(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }
}
