# Kyle Whitlatch CS5402


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
PATHTOFILES = 'C:/Users/Miner'


def main():

    #set up pandas
    outfile = open("output.txt","w+")
    train_df = pd.read_csv(PATHTOFILES + '/train.csv')
    test_df = pd.read_csv(PATHTOFILES + '/test.csv')
    combine = [train_df, test_df]
    #pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    #pd.set_option(display.width, 1000)

    #write tables out to files
    outfile.write(str(train_df))
    outfile.write("\n\n\n\n")
    outfile.write(str(test_df))
    outfile.write("\n\n\n\n")
    outfile.write(str(combine))
    outfile.write("\n\n\n\n")

    #print column headers
    print(list(train_df))
    print(list(test_df))

    #find values with null
    for c in train_df:
        for elements in train_df[c]:
            if 'nan' == str(elements):
                print(c)
                break
    for c in test_df:
        for elements in test_df[c]:
            if 'nan' == str(elements):
                print(c)
                break

    #print info on numeric
    numeric = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age']
    for n in numeric:
        print(train_df[n].describe())

    #print info on categorical
    for c in train_df:
        if str(c) not in numeric:
            print(train_df[c].describe())


    #print numerical correlations
    print(train_df.corr())

    #print if women were more likely to survive
    print(train_df.assign(Sex=train_df.Sex.astype('category').cat.codes).corr())

    #plot survived by age
    train_df.hist(column='Age', by='Survived')
    plt.show()

    #plot survived and class by age
    train_df.hist(column='Age', by=['Survived', 'Pclass'])
    plt.show()

    #plot fare and sex by embarked and survived
    train_df.hist(column=['Sex', 'Fare'], by=['Embarked', 'Survived'])
    plt.show()

    #check tickets for duplicates
    ticketcount = 0
    duplicates = 0
    for index, e in enumerate(train_df['Ticket']):
        for index, e2 in enumerate(train_df['Ticket']):
            ticketcount += 1
            if index != index and str(e) == str(e2):
                duplicates += 1
    print(float(duplicates)/float(ticketcount))

    countDupeCabin = 0
    for elements in train_df['Cabin']:
        if 'nan' == str(elements):
            countDupeCabin += 1
            break
    print(countDupeCabin)

    #check correlation on Cabin to survived
    print(train_df.assign(Cabin=train_df.Cabin.astype('category').cat.codes).corr())

    #make Sex into numerical category Gender
    train_df['Gender'] = train_df.apply(lambda row: 1 if row.Sex == 'female' else 0, axis=1)
    print(train_df.corr())

    #populate nan in Age column
    for e in train_df['Age']:
        if str(e) == 'nan':
            e = (str(train_df['Age'].mean()))
            print((str(train_df['Age'].mean())))

    #populate nan in Embarked column
    for e in train_df['Embarked']:
        if str(e) == 'nan':
            e = (str(train_df['Embarked'].mode()))
            print((str(train_df['Embarked'].mode())))

    #populate nan in Fare
    for e in train_df['Fare']:
        if str(e) == 'nan':
            e = (str(train_df['Fare'].mode(numeric_only=1)))
            print((str(train_df['Fare'].mode(numeric_only=1))))

    #make an ordinal Fare Column
    train_df['FareBand'] = train_df.apply(lambda row: 0, axis=1)
    for i,e in enumerate(train_df['Fare']):
        if e >= 0 and e <= 7.91:
            e = 0
        elif e > 7.91 and e <= 14.454:
            e = 1
        elif e > 14.454 and e <= 31:
            e = 2
        elif e > 31:
            e = 3

    print(train_df['Fare'])
if __name__ == '__main__':
    main()