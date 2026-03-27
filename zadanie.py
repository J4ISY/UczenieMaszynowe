
Wstecz

Nie przesłano

Prześlij
Zadanie 2
Termin wykonania dzisiaj o 13:30
Instrukcje
Używając funkcji describe() proszę uzupełnić analizę wykonaną w zadaniu 1 o opis danych statystycznych dla wybranych danych. Swoją analizę powiększcie o większą ilość cech ( kolumn) i zakończcie podsumowaniem - czy wybrany przez was dataset będzie nadawał się do użycia w modelu uczenia maszynowego? Tak, nie, dlaczego?
Przykładowy kod z dodatkową modyfikacją wykresów, zmianą ich kolorów, umiejscowieniem kilku wykresów na jednym 'obrazku' itd.


figure, ax = plt.subplots(2,2)


#creating scatter plot representing the connection between number of rooms and prices
ax1 = sub_data1.plot.scatter(x='Rooms', y='Price', ax=ax[0,0], color='yellow')
#plt.show() # showing plot

sub_data2 = data[['YearBuilt','Price']]
sub_data2 = sub_data2.dropna()
sub_data2.YearBuilt = sub_data2.YearBuilt.astype(int) # changing year type from float (1989.0) to int (1980)
#crating linear plot
sub_data2.plot(x='YearBuilt', y='Price', ax=ax[0,1], color='red')
#plt.show()

grouped = sub_data2.groupby('YearBuilt')['Price'].mean()# groupping values by year built and calculating mean price
grouped.plot(x='YearBuilt', y='Price', kind='bar', ax=ax[1,0]) #creating bar graph

#plt.show()

#saving graph
fig = ax1.get_figure()
fig.savefig('fig1.jpg')

#-------------------- Theory 2 code ----------------------------------------------------------------------------------


new_group = sub_data1.loc[sub_data1.Rooms > 2]

only1bathroom = data.loc[data.Bathroom == 1, ['Price']]
only2bathroom = data.loc[(data.Bathroom == 1) & (data.Car == 0), ['Price']]
#print(data.head(30))

values = data['Suburb'].unique()
values_df = pd.DataFrame(values, columns=['Suburb'])
j = 0
for i in values_df['Suburb']:
    data.loc[data['Suburb'] == i, 'id'] = j
    j = j+1

data2 = pd.read_csv("job_descriptions.csv")
experiences = data2['Experience'].unique()

experiences_df = pd.DataFrame(experiences, columns=['Experience'])

for i in experiences_df['Experience']:
    temp1 = i.split(' to ')
    temp2 = temp1[1].split(' ')
    experiences_df.loc[experiences_df['Experience'] == i, 'from'] = temp1[0]
    experiences_df.loc[experiences_df['Experience'] == i, 'to'] = temp2[0]


experiences_df['from'] = experiences_df['from'].astype(int)
experiences_df['to'] = experiences_df['to'].astype(int)

for i in experiences_df['Experience']:
    experiences_df.loc[experiences_df['Experience'] == i, 'num'] = experiences_df.loc[experiences_df['Experience'] == i,
    'to'] - experiences_df.loc[experiences_df['Experience'] == i, 'from']

#ax3 = experiences_df.plot(x='num', y='to', kind='bar', figsize=(10,5), color="red") #creating bar graph

ax2 = experiences_df.plot(x='num', y='from', kind='bar', figsize=(10,5), color="green") #creating bar graph


w, x = 0.4, np.arange(len(experiences_df['from']))


ax[1,1].bar(x - w/2, experiences_df['from'], width=w, label='from')
ax[1,1].bar(x + w/2, experiences_df['to'], width=w, label='to')

plt.show()

