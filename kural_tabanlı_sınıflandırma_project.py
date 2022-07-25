######################Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama############################
##İŞ PROBLEMİ##
"""Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona) oluşturmak ve bu yeni müşteri tanımlarına göre segmentler
oluşturup bu segmentlere göre yeni gelebilecek müşterilerin
şirkete ortalama ne kadar kazandırabileceğini tahmin etmek
istemektedir"""

##VERİ SETİ HİKAYESİ##
"""Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu
ürünleri satın alan kullanıcıların bazı demografik bilgilerini barındırmaktadır. Veri
seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı
tablo tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir
kullanıcı birden fazla alışveriş yapmış olabilir""""

##DEGİSKENLER##
#PRICE: Müşterinin harcama tutarı
#SOURCE: Müşterinin bağlandığı cihaz türü
#SEX: Müşterinin cinsiyeti
#COUNTRY: Müşterinin ülkesi
#AGE: Müşterinin yaşı


###########Uygulama Öncesi Veri Seti:
# -----------------------------------------
#  PRICE | SOURCE  | SEX  | COUNTRY | AGE
# ----------------------------------------
#     39 | android | male |  bra    |  17
#     39 | android | male |  bra    |  17
#     49 | android | male |  bra    |  17
#     29 | android | male |  tur    |  17
#     49 | android | male |  tur    |  17
# -----------------------------------------


################Hedeflenen çıktı:


# -----------------------------------------------
# |  customers_level_based   |  PRICE  | SEGMENT
# ---------------------------------------------
# | BRA_ANDROID_FEMALE_0_18  | 35.6453 |    B
# | BRA_ANDROID_FEMALE_19_23 | 34.0773 |    C
# | BRA_ANDROID_FEMALE_24_30 | 33.8639 |    C
# | BRA_ANDROID_FEMALE_31_40 | 34.8983 |    B
# | BRA_ANDROID_FEMALE_41_66 | 36.7371 |    A
# +--------------------------------------------


############# GÖREV 1 #################
#1.persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

import pandas as pd
import numpy as np
df=pd.read_csv("datasets/datasets_files/persona.csv")
df.head()
#2. Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

#3. Kaç unique PRICE vardır?
df["PRICE"].unique()

#4.Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

#5. Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()
#6. Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY")["PRICE"].sum()

#7. SOURCE türlerine göre satış sayıları nedir?
df.groupby("SOURCE")["PRICE"].value_counts()

#8. Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY")["PRICE"].mean()

#9. SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE")["PRICE"].mean()

#10.COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["SOURCE","COUNTRY"])["PRICE"].mean()

########## GOREV 2 #############
##COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#İSTENEN CIKTI
#COUNTRY |  SOURCE |   SEX  | AGE | PRICE  |
#  bra     andorid   female   15    38.71
#                             16    35.94
#                             17    35.66
#                             18    32.25
#                             19    35.25


df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})


############ GOREV 3 ###########
#Çıktıyı PRICE’a göre sıralayınız.
#Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uyguladık.
agg_df=df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE",ascending=False)
agg_df

############ GOREV 4 ###########
# Indekste yer alan isimleri değişken ismine çeviriniz.
agg_df=agg_df.reset_index()  #reset index fonksıyonu indekslenmeyı duzenler dıyebılırız

############ GOREV 5 ###########
#Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz
#Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'
myLabels = ["0_18", "19_23", "24_30", "31_40", "41_70"]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70], labels=myLabels)
print(agg_df.head())


############ GOREV  6 ###########
# Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: customers_level_based
agg_df["customer_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].apply("_".join, axis=1)
agg_df.customer_level_based.value_counts()
new_persona = agg_df.groupby("customer_level_based").agg({"PRICE": "mean"})
new_persona.reset_index(inplace=True)
new_persona.customer_level_based.value_counts()
new_persona["customer_level_based"] = new_persona.customer_level_based.apply(lambda x: x.upper())
new_persona.head()


############ GOREV  7 ###########

 # Yeni müşterileri (personaları) segmentlere ayırınız.
 # Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
 # Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
 # Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(10)
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})

############ GOREV  8 ###########
# Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user="tur_android_female_31_40"
agg_df[agg_df["customer_level_based"]==new_user]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
new_user="fra_ios_female_31_40"
agg_df[agg_df["customer_level_based"]==new_user]