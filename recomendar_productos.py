import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Cargar los datos desde el archivo CSV
data = pd.read_csv('SUPRISE/historial_compras.csv')

# Cargar el archivo de productos
productos_data = pd.read_csv('SUPRISE/datos_productos3_10.csv')

# Crear un objeto Reader para especificar el formato de los datos
reader = Reader(rating_scale=(1, 5))

# Cargar los datos en un objeto Dataset de Surprise
dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Dividir los datos en conjuntos de entrenamiento y prueba
trainset, testset = train_test_split(dataset, test_size=0.25)

# Crear y entrenar el modelo
model = SVD()
model.fit(trainset)

# Hacer predicciones en el conjunto de prueba
predictions = model.test(testset)

# Calcular la precisión del modelo
accuracy.rmse(predictions)

# Crear un diccionario para mapear item_id a nombre de producto
id_to_name = dict(zip(productos_data['item_id'], productos_data.iloc[:, 2]))

def get_recommendations_with_names(user_id, model, data, num_recommendations=5):
    # Obtener las recomendaciones de productos
    recommendations = get_recommendations(user_id, model, data, num_recommendations)
    
    # Convertir item_id a nombres de producto
    recommendations_with_names = [(id_to_name[item_id],item_id) for item_id in recommendations if item_id in id_to_name]
    
    return recommendations_with_names

def get_recommendations(user_id, model, data, num_recommendations=5):
    # Obtener todos los productos únicos
    all_items = data['item_id'].unique()
    
    # Obtener los productos que el usuario ya ha comprado
    user_items = data[data['user_id'] == user_id]['item_id']
    
    # Generar predicciones para productos no comprados
    predictions = []
    for item_id in all_items:
        if item_id not in user_items.values:
            # Predecir la calificación para el producto no comprado
            predicted_rating = model.predict(user_id, item_id).est
            predictions.append((item_id, predicted_rating))
    
    # Ordenar las predicciones por calificación descendente
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Seleccionar los mejores productos recomendados
    recommended_items = [item[0] for item in predictions[:num_recommendations]]
    
    return recommended_items

# Obtener y mostrar las recomendaciones para un usuario específico con nombres de productos
user_id = 1
recommendations_with_names = get_recommendations_with_names(user_id, model, data)
print(f'Top 5 recomendaciones para el usuario {user_id}: {recommendations_with_names}')







