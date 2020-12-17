import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

model = joblib.load('model.sav')
fn = ["Pclass", "Age", "Sex", "Embarked"]
cn = ["Survived", "Died"]
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
plot_tree(model,
          feature_names=fn,
          class_names=cn,
          filled=True,
          ax=axes)

data = pd.read_csv('train.csv')
fig2, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 14), dpi=300)
sns.countplot(data=data, hue='Sex', x='Survived', ax=axes[0])
sns.countplot(data=data, hue='Pclass', x='Survived', ax=axes[1])
sns.countplot(data=data, hue='Embarked', x='Survived', ax=axes[2])
sns.boxplot(data=data, x='Survived', y='Age', ax=axes[3])

st.title("Титаник")
st.text("Для демонстрации работы проекта было выбрано соревнование Титаник с Kaggle\nhttps://www.kaggle.com/c/titanic")
st.text("""Потопление "Титаника" - одно из самых печально известных кораблекрушений в истории.
15 апреля 1912 года, во время своего первого плавания, широко известный “непотопляемый”
"Титаник" затонул после столкновения с айсбергом. К сожалению, спасательных шлюпок
не хватило на всех, кто находился на борту, в результате чего погибли 1502 человека
из 2224 пассажиров и членов экипажа. Хотя в выживании и был определенный элемент удачи,
похоже, что некоторые группы людей имели больше шансов выжить, чем другие.""")
st.text("В ходе работы необходимо предсказать вероятность того, что пассажир выжил")
st.subheader("Отбор признаков")
st.text("Для построения модели были выбраны следующие признаки: Pclass, Age, Sex, Embarked")
st.text("Ниже представлены распределения выживших и нет по этим признакам")
st.pyplot(fig2)
st.subheader("Построение модели")
st.text("Для бейзлайна модели было выбрано решающее дерево, ниже представлена его визуализация")
st.pyplot(fig)
st.markdown("""*Теперь самое интересное.\nВыберите характеристики для пассажира вероятность остаться
            в живых которого вы бы хотели оценить*""")
age = st.slider('Возарст', min_value=1, max_value=120, value=27)
sex = st.radio('Пол', ['Мужской', 'Женский'])
pclass = st.radio('Класс билета', ['1', '2', '3'])
embarked = st.radio('Порт посадки', ['Southampton', 'Cherbourg', 'Queenstown'])
sex_dict = {'Мужской': 1, 'Женский': 0}
embarked_dict = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}
predict_dict = {
    'Pclass': [int(pclass)],
    'Age': [int(age)],
    'Sex': [sex_dict[sex]],
    'Embarked': [embarked_dict[embarked]]
}
surv_prob = model.predict_proba(pd.DataFrame.from_dict(predict_dict))[:, 1][0]
st.write(f"Ваш пассажир отправился из порта {embarked} в возрасте {age} лет, пол - {sex}, класс билета - {pclass}")
st.write(f"Вероятность того, что он выживет равна {surv_prob:.2f}")
if surv_prob < 0.5:
    image = Image.open('images/died.jpg')
else:
    image = Image.open('images/surv.jpg')
st.image(image, use_column_width=True)
