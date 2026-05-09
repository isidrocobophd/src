import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(csv_path):
    print("="*80)
    print("FASE 1: CARGA Y PREPROCESAMIENTO DE DATOS")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    print(f"\nDatos originales cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    target_col = df.pop('Target')
    df.insert(0, 'Target', target_col)
    print(f"Columna Target movida a primera posicion")
    
    df_filtered = df[df['Target'] != 'Enrolled'].copy()
    print(f"Instancias Enrolled eliminadas: {df.shape[0] - df_filtered.shape[0]}")
    print(f"Datos despues del filtrado: {df_filtered.shape[0]} filas")
    print(f"Distribucion de Target:\n{df_filtered['Target'].value_counts()}")
    
    return df_filtered

def split_data(df):
    print("\n" + "="*80)
    print("FASE 2: DIVISION DE DATOS (80-10-10)")
    print("="*80)
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    y_encoded = (y == 'Graduate').astype(int)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nEntrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"Validacion: {X_val.shape[0]} muestras ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"Prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def correlation_analysis(X_train, y_train):
    print("\n" + "="*80)
    print("FASE 3 & 4: ANALISIS DE CORRELACION Y SELECCION DE ATRIBUTOS")
    print("="*80)
    
    df_corr = X_train.copy()
    df_corr['Target'] = y_train
    correlations = df_corr.corr()['Target'].drop('Target').abs()
    
    print(f"\nTotal de atributos originales: {len(correlations)}")
    
    strong_features = correlations[correlations >= 0.2].index.tolist()
    weak_features = correlations[correlations < 0.2].index.tolist()
    
    print(f"Atributos eliminados (|r| < 0.2): {len(weak_features)}")
    print(f"Atributos mantenidos (|r| >= 0.2): {len(strong_features)}")
    
    print(f"\nTop 10 atributos por correlacion:")
    for feature, corr_val in correlations.sort_values(ascending=False).head(10).items():
        print(f"  - {feature}: {corr_val:.4f}")
    
    return strong_features

def train_models(X_train, X_val, X_test, y_train, y_val, y_test, selected_features):
    print("\n" + "="*80)
    print("FASE 5: ENTRENAMIENTO DE MODELOS")
    print("="*80)
    
    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    X_test_sel = X_test[selected_features]
    
    print(f"\nModelo entrenado con {len(selected_features)} atributos seleccionados")
    
    print("\n📊 LOGISTIC REGRESSION")
    lr = LogisticRegression(
        penalty='l2', dual=False, tol=0.0001, C=1.0,
        fit_intercept=True, intercept_scaling=1, class_weight=None,
        random_state=None, solver='newton-cholesky', max_iter=100,
        multi_class='auto', verbose=0, warm_start=False,
        n_jobs=None, l1_ratio=None
    )
    lr.fit(X_train_sel, y_train)
    print("Modelo entrenado")
    
    print("\n🌳 DECISION TREE")
    dt = DecisionTreeClassifier(
        criterion='entropy', splitter='best', max_depth=5,
        min_samples_split=150, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None,
        random_state=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0
    )
    dt.fit(X_train_sel, y_train)
    print("Modelo entrenado")
    
    return lr, dt, X_train_sel, X_val_sel, X_test_sel

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    results = {}
    for phase, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        results[phase] = {
            'precision': precision_score(y, y_pred, zero_division=0),
            'accuracy': accuracy_score(y, y_pred),
            'recall': recall_score(y, y_pred, zero_division=0)
        }
    return results

def print_results(lr_results, dt_results):
    print("\n" + "="*80)
    print("FASE 6: COMPARACION DE RESULTADOS")
    print("="*80)
    
    metrics = ['precision', 'accuracy', 'recall']
    phases = ['train', 'val', 'test']
    phase_names = {'train': 'ENTRENAMIENTO', 'val': 'VALIDACION', 'test': 'PRUEBA'}
    metric_names = {'precision': 'Precision', 'accuracy': 'Exactitud', 'recall': 'Exhaustividad'}
    
    for phase in phases:
        print(f"\n{'-'*80}")
        print(f"{phase_names[phase]}")
        print(f"{'-'*80}")
        print(f"{'Metrica':<20} {'Logistic Reg.':<20} {'Decision Tree':<20} {'Diferencia':<15}")
        print(f"{'-'*80}")
        
        for metric in metrics:
            lr_val = lr_results[phase][metric]
            dt_val = dt_results[phase][metric]
            diff = lr_val - dt_val
            winner = "LR" if diff > 0 else ("DT" if diff < 0 else "IGUALES")
            print(f"{metric_names[metric]:<20} {lr_val:<20.4f} {dt_val:<20.4f} {diff:+.4f} {winner}")
    
    print(f"\n{'='*80}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*80}")
    print("\nMejor rendimiento por metrica (promedio en todas las fases):")
    
    for metric in metrics:
        lr_avg = np.mean([lr_results[p][metric] for p in phases])
        dt_avg = np.mean([dt_results[p][metric] for p in phases])
        winner = "Logistic Regression" if lr_avg > dt_avg else ("Decision Tree" if dt_avg > lr_avg else "Empate")
        print(f"\n{metric_names[metric]}:")
        print(f"  - Logistic Regression: {lr_avg:.4f}")
        print(f"  - Decision Tree: {dt_avg:.4f}")
        print(f"  - Ganador: {winner}")

def main():
    print("\n╔" + "="*78 + "╗")
    print("║" + " RETO ML3: CLASIFICADOR DE ESTUDIANTES (GRADUADOS/ABANDONADOS)".center(78) + "║")
    print("║" + " Analisis Comparativo: Logistic Regression vs Decision Tree".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        df = load_and_preprocess_data('data.csv')
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        selected_features = correlation_analysis(X_train, y_train)
        lr, dt, X_train_sel, X_val_sel, X_test_sel = train_models(
            X_train, X_val, X_test, y_train, y_val, y_test, selected_features
        )
        
        print("\n" + "="*80)
        print("EVALUACION DE MODELOS")
        print("="*80)
        
        lr_results = evaluate_model(lr, X_train_sel, X_val_sel, X_test_sel, y_train, y_val, y_test)
        dt_results = evaluate_model(dt, X_train_sel, X_val_sel, X_test_sel, y_train, y_val, y_test)
        
        print_results(lr_results, dt_results)
        
        print("\n" + "="*80)
        print("ANALISIS COMPLETADO EXITOSAMENTE")
        print("="*80 + "\n")
        
    except FileNotFoundError:
        print("\nError: No se encontro el archivo data.csv")
        print("Descargalo desde: https://www.kaggle.com/datasets/mahwiz/students-dropout-and-academic-success-dataset\n")
    except Exception as e:
        print(f"\nError: {str(e)}\n")
        raise

if __name__ == "__main__":
    main()
