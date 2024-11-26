print("-"*45)
print( 
"""
İlk Python Hesap Makinesi Uygulamama Hoşgeldin!\nBir çok hesabı tek tuşla yapabilirsin!
Lütfen istediğin işlemi seç\nÇıkmak için 'q'ya bas!
1)Toplama 2)Çıkarma 3)Çarpma 4)Bölme\n5)Üs alma 6)Mod alma 7)Daire Çevre-Alan Hesabı\n8)Dikdörtgen Çevre-Alan Hesabı\n9)Mevduat Faiz Hesabı 'Yıllık'\n10)Günlük Faiz Hesabı
11)Anüite gelecek değer hesabı\n12)Senet faiz hesabı\n13)Gerilme ve Emniyet Gerilmesi Hesabı\n14)Hicri-Miladi Yıl Dönüşümü\n15)Rumi-Miladi Yıl Dönüşümü\n16)Miladi-Rumi-Hicri Yıl Dönüşümü
"""
)
print("-"*45)

secim = 0

def MiladidenÇevir():
    while True:
        try:
            secim = input("Miladiden çevirmek istediğiniz yılı seçiniz: (Hicri/Rumi) ").lower()
            miladi_Yil = int(input("Miladi yılı giriniz: "))
            if miladi_Yil <= 0:
                print("Lütfen pozitif bir yıl değeri giriniz!")
                continue
            if secim == "rumi":
                Miladiden_RumiYıl = miladi_Yil - 584
                return Miladiden_RumiYıl
            elif secim == "hicri":
                de = (miladi_Yil - 621) // 33
                Miladiden_HicriYıl = de + (miladi_Yil - 621)
                return Miladiden_HicriYıl
            else:
                print("Lütfen 'Hicri' veya 'Rumi' seçeneklerinden birini giriniz.")
        
        except ValueError:
            print("Uygun bir değer giriniz!")




def RumidenMiladiyeÇevir():
    while True:
        try:
            print("-" * 35)
            rumiYil = int(input("Rumi yılı giriniz: "))
            print("-" * 35)
            if rumiYil <= 0:
                print("Lütfen pozitif bir Rumi yıl giriniz.")
                continue
            Rumiden_MiladiYil = rumiYil + 584
            return Rumiden_MiladiYil
        except ValueError:
            print("Geçerli bir sayı giriniz")


def HicridenMiladiyeÇeviri():
    while True:
        try:
            print("-"*35)
            hicriYil = int(input("Hicri yılı giriniz: "))
            print("-"*35)
            if hicriYil <= 0:
                print("Lütfen pozitif bir Hicri yıl giriniz.")
                continue
            ab = hicriYil // 33
            cd = hicriYil - ab
            miladiYil = cd + 622
            return miladiYil
        except ValueError:
            print("Geçerli bir tam sayı giriniz.")


 
def GelecekDegerHesapla():
        while True:
            try:
                print("-"*35)
                esitOdeme = float(input("Eşit ödeme miktarı: "))
                faizOrani = float(input("Yıllk faiz oranı: "))
                devreSayisi = float(input("Ödeme yapilan yıl"))
                print("-"*35)

                if(esitOdeme <=0 ):
                    print("Lütfen pozitif değerler giriniz! ")
                    continue
                elif(esitOdeme == "q"):
                     break
                
                sonuc =  esitOdeme * ((1 + faizOrani) ** (devreSayisi) - 1) / faizOrani
                return sonuc
            except ValueError:
                 print("Lütfen geçerli bir değer giriniz! ")

def GerilmeHesapla():
    while True:
        try:
            print("-"*35)
            yuk = float(input("Yük miktarı kg olarak giriniz: "))
            alan = float(input("Alan giriniz (cm2):  "))
            gerilme = yuk / alan
            print("-"*35)
            birim=input("Birim dönüşüm  işlemi yapmak istiyorsanız birimi seçiniz: (kg/cm2,Kpa,Mpa,Gpa)")
            match birim:
                case "kg/cm2":
                    return gerilme,"kg/cm2"
                case "Kpa"|"kpa":
                    return gerilme*10,"Kpa"
                   
                case "Mpa"|"mpa":
                    return gerilme*0.1,"Mpa"
                    
                case "Gpa"|"gpa":
                    return gerilme*0.0001,"Gpa"
                    
                case _:
                    print("Lütfen geçerli bir birim giriniz! ")
                                       
        except ValueError:
            print("Lütfen geçerli bir değer giriniz! ")
     

while True:
    islem = input("İslem kodunu giriniz:")
    
    if(islem == "q"):
        print("Yine bekleriz!")
        break
    
    elif islem not in ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"]:
        print("Tanımsız işlem,Tekrar deneyin")
        continue
    
   
    if islem == "7":
        yaricap = float(input("Lütfen daire yarıçapını giriniz: "))      
        pi = 3.1459
        try:
            print("Daire alanı: ",pi*(yaricap**2))
            print("Daire çevresi: ",2*pi*yaricap)
        except ValueError:
            print("Geçerli bir çap değeri giriniz! ")
            continue
        
    elif islem == "9":
        anapara = float(input("Yatırılan anapara miktarı: "))
        faizOrani = float(input("Yıllk faiz oranını giriniz(Ör: 0.5): "))
        zaman = int(input("Paranın kalacağı gün: "))
        
        try:
            print("{} TL'nin {} faiz oranı ile {} süreyle alacağız faiz: ".format(anapara,faizOrani,zaman),(anapara*faizOrani*zaman)/36500)
        except ValueError:
            print("Uygun bir değer giriniz! ") 
            continue

    elif islem == "10":
        anapara = float(input("Yatırılan anapara miktarı: "))
        faizOrani = float(input("Günlük faiz oranını giriniz(Ör:0.5): "))
        zaman = float(input("Paranın kalacağı gün: "))

        try:
            print("{} TL'nin {} faiz oranı ile {} süreyle alacağız faiz:  TL".format(anapara,faizOrani,zaman),(anapara*faizOrani*zaman)/36500 )
        except ValueError:
            print("Uygun bir değer giriniz! ") 
            continue

    elif islem == "11":      
            sonuc = GelecekDegerHesapla()
            print(f"Hesaplanan toplam para: {sonuc:.2f} TL")      

    elif islem == "12":
            faizOrani = float(input("Faiz oranını girin (örnek: 0.15): "))
            anapara = float(input("Anaparayı girin: "))
            vade_gun = int(input("Vade gün sayısını girin: "))
            
            SenetHesapla = lambda faizOrani, anapara, vade_gun: (anapara * faizOrani) * vade_gun / 360
            sonuc = SenetHesapla(faizOrani, anapara, vade_gun)
            print(f"Senet faizi: {sonuc:.2f} TL")

    elif islem == "13":
              gerilme,birim = GerilmeHesapla()
              print("Hesaplanan Gerilme: {} {} ".format(gerilme,birim))
              


    elif islem == "14":
                    miladiYil = HicridenMiladiyeÇeviri()
                    print(f"Girilen Hicri yılın karşılığı: {miladiYil} ")


    elif islem =="15":
        Rumiden_MiladiYil = RumidenMiladiyeÇevir()
        print(f" Girilen Rumi yılın miladi yıl karşılığı: {Rumiden_MiladiYil}")

    
    elif islem == "16":
        miladiYil = MiladidenÇevir()
        secim = input("Çevrilen yılı görmek istediğiniz takvim sistemini seçiniz: (Hicri/Rumi) ").strip().lower()
        if secim == "hicri":
            print(f"Girilen Miladi yılın Hicri yıl karşılığı: {miladiYil}")
        elif secim == "rumi":
             print(f"Girilen Miladi yılın Rumi yıl karşılığı: {miladiYil}")


   
                
    else: 
        try:
            sayi1=float(input("İlk sayiyi yada cm cinsinden uzun kenarı girin: "))
            sayi2 = float(input("İkinci sayiyi yada cm cinsinden kısa kenarı girin: "))
        except ValueError:
            print("Geçerli bir değer giriniz! ")
            continue
        
    
    if islem == "1":
        print(sayi1, "+", sayi2, "=", sayi1 + sayi2)
    elif islem == "2":
        print(sayi1, "-", sayi2, "=", sayi1 - sayi2)
    elif islem == "3":
        print(sayi1, "X", sayi2, "=", sayi1 * sayi2)
    elif islem == "4":
        try:
            print(sayi1, "/", sayi2, "=", sayi1 / sayi2)
        except ZeroDivisionError:
            print("Bir sayi 0'a bölünemez!Tekrar deneyin! ")
            continue
        
    elif islem == "5":
        print("{} üssü {}: ".format(sayi1, sayi2), sayi1 ** sayi2)
    elif islem == "6":
        try:
            print("{} mod {}: ".format(sayi1,sayi2),sayi1 % sayi2)
        except ZeroDivisionError:
            print("Mod almada bölen 0 olamaz,tekrar deneyin!")
            continue 

    elif islem == "8":
        try:
            print("Kenarları {} ile {} olan dikdörtgenin çevresi: ".format(sayi1,sayi2),2*(sayi1 + sayi2))
            print(f"Kenarları {sayi1} ile {sayi2} olan dikdörtgenin alanı:\t{sayi1*sayi2} cm ")
        except ZeroDivisionError:
            print("Kenar değeri 0 olamaz!Tekrar deneyin!")
            continue

    
    devam = input("Baska bir islem yapmak ister misiniz?(e/h)")
    if(devam == "h" or devam == "H"):
        print("Teşekkürler yine bekleriz!")
        break
       
     
