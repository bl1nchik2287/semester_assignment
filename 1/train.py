import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class RainForecastModel:
    def __init__(self, csv_file="weatherHistory.csv"):
        self.csv_file = csv_file
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def _load_dataset(self):
        df = pd.read_csv(self.csv_file)
        df['Rain'] = (df['Precip Type'] == 'rain').astype(int)

        features = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
                    'Pressure (millibars)']
        X = df[features].values
        y = df['Rain'].values

        return X, y

    def _prepare_data(self, test_ratio=0.3):
        X, y = self._load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=42
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def _construct_model(self, input_dim):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, epochs=150, batch_size=32, model_path="weather_model.h5"):
        X_train, X_test, y_train, y_test = self._prepare_data()
        self.model = self._construct_model(X_train.shape[1])

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        self.model.save(model_path)
        print(f"\nМодель сохранена как: {model_path}")

    def evaluate_performance(self):
        _, X_test, _, y_test = self._prepare_data()
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"\nРезультат на тестовой выборке: {test_accuracy * 100:.2f}%")

        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        conf_matrix = confusion_matrix(y_test, y_pred)

        self._plot_confusion(conf_matrix)

        print("\nОтчет по метрикам:")
        print(classification_report(y_test, y_pred, target_names=["Без осадков", "Дождь"]))
#Кривые
    def show_training_curves(self):
        if not self.history:
            print("История обучения отсутствует.")
            return
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Обучение')
        plt.plot(self.history.history['val_accuracy'], label='Валидация')
        plt.title('Точность модели')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()

        #Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Обучение')
        plt.plot(self.history.history['val_loss'], label='Валидация')
        plt.title('Потери модели')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()

        plt.tight_layout()
        plt.show()
#Матрица ошибок
    def _plot_confusion(self, matrix):
        plt.figure(figsize=(6, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Без осадков", "Дождь"],
                    yticklabels=["Без осадков", "Дождь"])
        plt.title("Матрица ошибок")
        plt.xlabel("Предсказано")
        plt.ylabel("Фактически")
        plt.show()


def run_pipeline():
    print("Запуск системы предсказания осадков...")

    model = RainForecastModel()

    print("\n[1] Обучение модели")
    model.fit()

    print("\n[2] Проверка точности")
    model.evaluate_performance()

    print("\n[3] Визуализация результатов обучения")
    model.show_training_curves()

    print("\nГотово!")


if __name__ == "__main__":
    run_pipeline()
#На этом датасете показывает точность 98% (и более), но мб это датасет такой несбалансированный