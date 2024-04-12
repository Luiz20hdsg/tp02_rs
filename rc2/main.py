import pandas as pd

class CustomizedRocchio:

  def __init__(self, ratings_df):
    # Inicializa a classe Rocchio com o dataframe de avaliações.
    self.__user_ratings = ratings_df[['UserId', 'ItemId', 'Rating']].copy()
    
    # Redimensiona as notas para um novo intervalo.
    self.__user_ratings['Rating'] = self.__user_ratings.apply(lambda row: self.__rescale_rating(row['Rating']), axis=1)

  def fit(self, features_df):
    # Ajusta a classe Rocchio com um dataframe de características.
    self.__item_features = self.__user_ratings.copy()

    # Realiza a vetorização das características.
    self.__vectorize_features(features_df)
    
    # Calcula a matriz de utilidade do usuário.
    self.__compute_user_utility()

  def predict(self, user_item_df):
    # Realiza previsões para pares de usuário e item.
    targets = user_item_df[['ItemId', 'UserId']].copy()
    targets = targets.merge(self.__item_features, on='ItemId', how='outer')
    targets = targets.fillna(0)

    targets = pd.merge(targets, self.__user_utility, on='UserId')

    # Calcula a similaridade do cosseno entre usuário e item.
    cosine_similarity = targets[['UserId', 'ItemId']]
    cosine_similarity['user_DOT_item'] = 0
    cosine_similarity['user_distance'] = 0
    cosine_similarity['item_distance'] = 0

    for feature in self.__features_list:
      cosine_similarity['user_DOT_item'] += targets['u_' + feature] * targets[feature]
      cosine_similarity['user_distance'] += targets['u_' + feature] * targets['u_' + feature]
      cosine_similarity['item_distance'] += targets[feature]        * targets[feature]

    cosine_similarity['user_distance'] = cosine_similarity['user_distance'].pow(1./2)
    cosine_similarity['item_distance'] = cosine_similarity['item_distance'].pow(1./2)
    cosine_similarity['Similarity']    = cosine_similarity['user_DOT_item'] / \
                                         (cosine_similarity['user_distance'] * cosine_similarity['item_distance'])

    cosine_similarity = cosine_similarity.drop(['user_DOT_item', 'user_distance', 'item_distance'], axis=1)

    return cosine_similarity

  def __vectorize_features(self, features_df):
    # Realiza a vetorização das características.
    self.__item_features = features_df[['ItemId', 'Features']].copy()
    splitted_features = self.__item_features['Features'].str.split(' ')
    self.__features_list = list(dict.fromkeys([y for x in splitted_features for y in x]).keys())

    for feat in self.__features_list:
      self.__item_features[feat] = self.__item_features['Features'].apply(lambda x: 1 if feat in x else 0)

    self.__item_features = self.__item_features.drop(['Features'], axis=1)

  def __rescale_rating(self, old_rating):
    # Redimensiona as notas para um novo intervalo.
    if old_rating == 10:
        return 3
    elif old_rating > 7:
        return 2
    elif old_rating > 5:
        return 1
    return 0

  def __compute_user_utility(self):
    # Calcula a matriz de utilidade do usuário.
    self.__user_utility = self.__user_ratings.copy()
    self.__user_utility = self.__user_utility.merge(self.__item_features, on='ItemId')

    self.__user_utility['Frequency'] = self.__user_utility.groupby('UserId')['UserId'].transform('count')

    for feat in self.__features_list:
        self.__user_utility[feat] = self.__user_utility[feat] * self.__user_utility['Rating'] / self.__user_utility['Frequency']

    self.__user_utility = self.__user_utility.drop(['ItemId', 'Frequency', 'Rating'], axis=1)
    self.__user_utility = self.__user_utility.groupby('UserId').sum()

    for feat in self.__features_list:
        self.__user_utility['u_' + feat] = self.__user_utility[feat]
        self.__user_utility = self.__user_utility.drop(feat, axis=1)


import sys
import pandas as pd


# Função para ordenar as recomendações com base na similaridade e outras informações
def sort_recommendations(content, similarities_df):
    sorted_recommendations = similarities_df.merge(content[['ItemId', 'imdbVotes']], on='ItemId')
    sorted_recommendations['Similarity'] = sorted_recommendations['Similarity'] * sorted_recommendations['imdbVotes']
    fields_to_be_sorted = ['UserId', 'Similarity', 'imdbVotes']
    sorting_order_ascending = [True, False, False]
    sorted_recommendations = sorted_recommendations.sort_values(fields_to_be_sorted, ascending=sorting_order_ascending)
    return sorted_recommendations

# Função para imprimir as recomendações ordenadas
def recommendations(sorted_recommendations):
    output_file = open('submission.csv', 'w')
    print('UserId,ItemId', file=output_file)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    for row in sorted_recommendations.itertuples():
        print('{},{}'.format(row.UserId, row.ItemId), file=output_file)
    output_file.close()

# Função principal
def main():
    # Carrega os DataFrames de avaliações, informações dos itens e alvos
    ratings = pd.read_json(sys.argv[1], lines=True)
    content = pd.read_json(sys.argv[2], lines=True)
    targets = pd.read_csv(sys.argv[3], encoding='latin-1', sep=',')
    # Load data
    #ratings = pd.read_json('ratings.jsonl', lines=True)
    #content = pd.read_json('content.jsonl', lines=True)
    #targets = pd.read_csv('targets.csv')

    # Inicializa o modelo Rocchio com as avaliações
    rocchio = CustomizedRocchio(ratings)

    # Pré-processamento das features
    content['Genre'] = content['Genre'].map(lambda x: x.lower().replace(', ', ' '))
    content['Language'] = content['Language'].map(lambda x: x.lower().replace(', ', ' '))
    content['Features'] = content['Genre']
    content = content.drop(['Genre'], axis=1)

    # Tratamento dos campos utilizados para desempate
    content['imdbVotes'] = content['imdbVotes'].str.replace('N/A', '1', regex=True)
    content['imdbVotes'] = content['imdbVotes'].str.replace(',', '', regex=True)
    content['imdbVotes'] = content['imdbVotes'].astype(str).astype('float64')
    content['imdbVotes'] = content['imdbVotes'].fillna(0.0)

    content['imdbRating'] = content['imdbRating'].str.replace('N/A', '1', regex=True)
    content['imdbRating'] = content['imdbRating'].str.replace(',', '', regex=True)
    content['imdbRating'] = content['imdbRating'].astype(str).astype('float64')
    content['imdbRating'] = content['imdbRating'].fillna(0.0)

    # Ajusta o modelo Rocchio com as features do conteúdo
    rocchio.fit(content[['ItemId', 'Features', 'imdbRating', 'imdbVotes', 'Language', 'Year']])
    rocchio_similarities = rocchio.predict(targets)

    # Ordena as recomendações e imprime o resultado
    sorted_recommendations = sort_recommendations(content, rocchio_similarities)
    recommendations(sorted_recommendations)

if __name__ == '__main__':
    main()
