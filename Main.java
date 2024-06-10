import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        // Datos de batch size
        double[] x = {108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89};
        // Datos de machine efficiency
        double[] y = {95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93};

        // Modelos de regresión polinómica
        PolynomialRegression linearRegression = new PolynomialRegression(1);
        PolynomialRegression quadraticRegression = new PolynomialRegression(2);
        PolynomialRegression cubicRegression = new PolynomialRegression(3);

        // Ajustar modelos
        linearRegression.fit(x, y);
        quadraticRegression.fit(x, y);
        cubicRegression.fit(x, y);

        // Mostrar los coeficientes calculados
        System.out.println("Coeficientes de la regresión lineal: " + Arrays.toString(linearRegression.getCoefficients()));
        System.out.println("Coeficientes de la regresión cuadrática: " + Arrays.toString(quadraticRegression.getCoefficients()));
        System.out.println("Coeficientes de la regresión cúbica: " + Arrays.toString(cubicRegression.getCoefficients()));

        // Realizar predicciones
        double[] testValues = {50, 60, 70, 80, 90};
        System.out.println("Predicciones para la regresión lineal:");
        for (double value : testValues) {
            System.out.println("x = " + value + ", y = " + linearRegression.predict(value));
        }
        System.out.println("Predicciones para la regresión cuadrática:");
        for (double value : testValues) {
            System.out.println("x = " + value + ", y = " + quadraticRegression.predict(value));
        }
        System.out.println("Predicciones para la regresión cúbica:");
        for (double value : testValues) {
            System.out.println("x = " + value + ", y = " + cubicRegression.predict(value));
        }

        // Calcular e imprimir los coeficientes de correlación y determinación
        System.out.println("Coeficiente de determinación para la regresión lineal: " + linearRegression.r2(x, y));
        System.out.println("Coeficiente de determinación para la regresión cuadrática: " + quadraticRegression.r2(x, y));
        System.out.println("Coeficiente de determinación para la regresión cúbica: " + cubicRegression.r2(x, y));
    }
}