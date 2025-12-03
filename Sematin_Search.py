# %%
#!pip install openai scipy pandas

# %%
# OpenAI Settings
APIKey = ''

import openai
openai.api_key = APIKey


# %%
# Baza Wiedzy - Elektronika (54 produkty)
# Różnorodne kategorie dla ciekawych testów semantic search

knowledgebase = [
    # === LAPTOPY (10) ===
    "Laptop Dell XPS 13 9320 - procesor Intel Core i7-1260P 12-gen, 16GB RAM LPDDR5, dysk 512GB SSD NVMe, ekran 13.4 cala Full HD+ IPS dotykowy, Intel Iris Xe Graphics, waga 1.27kg, Windows 11 Pro, bateria do 12h, aluminiowa obudowa, cena 5499 zł",
    
    "Laptop Lenovo ThinkPad X1 Carbon Gen 10 - procesor Intel Core i5-1235U, 8GB RAM DDR4, dysk 256GB SSD, ekran 14 cali Full HD IPS matowy, Intel Iris Xe Graphics, waga 1.12kg, Windows 11, klawiatura z podświetleniem, czytnik linii papilarnych, bateria 15h, idealny do biura, cena 4299 zł",
    
    "Laptop HP Pavilion Gaming 15 - procesor AMD Ryzen 5 5600H, 16GB RAM DDR4, dysk 512GB SSD, ekran 15.6 cala Full HD IPS 144Hz, karta graficzna NVIDIA GeForce RTX 3050 4GB, podświetlana klawiatura RGB, chłodzenie Omen, Windows 11, waga 2.2kg, cena 3799 zł",
    
    "Laptop Asus ROG Strix G15 - procesor AMD Ryzen 7 5800H, 32GB RAM DDR4, dysk 1TB SSD NVMe, ekran 15.6 cala Full HD IPS 300Hz, NVIDIA GeForce RTX 3070 8GB, klawiatura mechaniczna RGB per-key, system chłodzenia ROG Intelligent Cooling, Windows 11, waga 2.3kg, dla wymagających graczy, cena 6999 zł",
    
    "Laptop Apple MacBook Air M2 - chip Apple M2 8-core CPU 8-core GPU, 8GB RAM unified memory, dysk 256GB SSD, ekran 13.6 cala Liquid Retina 2560x1664, kamera Full HD, głośniki przestrzenne, Touch ID, macOS Ventura, waga 1.24kg, bateria do 18h, bezszelestna praca, cena 5799 zł",
    
    "Laptop Acer Aspire 5 A515 - procesor Intel Core i3-1115G4, 8GB RAM DDR4, dysk 256GB SSD, ekran 15.6 cala Full HD IPS, Intel UHD Graphics, klawiatura numeryczna, Windows 11 Home, waga 1.8kg, bateria 7h, budżetowy laptop do nauki i pracy biurowej, cena 2299 zł",
    
    "Laptop MSI Creator Z16P - procesor Intel Core i9-12900H, 32GB RAM DDR5, dysk 2TB SSD NVMe, ekran 16 cali QHD+ 165Hz 100% DCI-P3, NVIDIA GeForce RTX 3080 Ti 16GB, kalibracja kolorów CalMAN, Thunderbolt 4, Windows 11 Pro, dla twórców wideo i grafików, waga 2.4kg, cena 12999 zł",
    
    "Laptop Huawei MateBook D15 - procesor Intel Core i5-1135G7, 8GB RAM DDR4, dysk 512GB SSD, ekran 15.6 cala Full HD IPS, Intel Iris Xe Graphics, smukła metalowa obudowa, kamera pop-up, Windows 11, bateria 10h, dobry stosunek ceny do możliwości, waga 1.56kg, cena 2799 zł",
    
    "Laptop Microsoft Surface Laptop 5 - procesor Intel Core i7-1255U, 16GB RAM LPDDR5x, dysk 512GB SSD, ekran dotykowy 13.5 cala PixelSense 2256x1504, Intel Iris Xe Graphics, Windows 11, kamera Windows Hello, elegancka obudowa Alcantara, waga 1.29kg, bateria 18h, premium ultrabook, cena 6499 zł",
    
    "Laptop Lenovo IdeaPad 3 15 - procesor AMD Ryzen 3 5300U, 8GB RAM DDR4, dysk 256GB SSD, ekran 15.6 cala Full HD TN, AMD Radeon Graphics, klawiatura numeryczna, Windows 11 Home S, waga 1.65kg, bateria 8h, najtańszy laptop dla studentów i podstawowego użytku, cena 1899 zł",
    
    # === TELEFONY (8) ===
    "Smartfon Apple iPhone 15 Pro Max - procesor Apple A17 Pro 3nm, 8GB RAM, pamięć 256GB, ekran 6.7 cala Super Retina XDR OLED 120Hz ProMotion, aparat 48MP główny + 12MP ultra-wide + 12MP tele 5x, nagrywanie ProRes 4K, tytanowa obudowa, USB-C, iOS 17, bateria 4422mAh, cena 6799 zł",
    
    "Smartfon Samsung Galaxy S24 Ultra - procesor Qualcomm Snapdragon 8 Gen 3, 12GB RAM, pamięć 512GB, ekran 6.8 cala Dynamic AMOLED 2X 120Hz QHD+, aparat 200MP główny + 50MP tele 5x + 10MP tele 3x + 12MP ultra-wide, rysik S Pen, IP68, Android 14, bateria 5000mAh ładowanie 45W, cena 6499 zł",
    
    "Smartfon Google Pixel 8 Pro - procesor Google Tensor G3, 12GB RAM, pamięć 256GB, ekran 6.7 cala LTPO OLED 120Hz, aparat 50MP główny z AI, nagrywanie 4K60 HDR, czysty Android 14 z 7 lat aktualizacji, funkcje AI Magic Editor i Best Take, bateria 5050mAh ładowanie 30W, cena 4999 zł",
    
    "Smartfon Xiaomi 13T Pro - procesor MediaTek Dimensity 9200+, 12GB RAM, pamięć 512GB, ekran 6.67 cala AMOLED 144Hz, aparat Leica 50MP główny + 50MP tele 2x + 12MP ultra-wide, ładowanie 120W HyperCharge (pełne w 19 min), Android 13 MIUI 14, bateria 5000mAh, dobry stosunek ceny do możliwości, cena 3299 zł",
    
    "Smartfon OnePlus 11 - procesor Qualcomm Snapdragon 8 Gen 2, 16GB RAM, pamięć 256GB, ekran 6.7 cala Fluid AMOLED 120Hz, aparat Hasselblad 50MP główny + 48MP ultra-wide + 32MP tele 2x, ładowanie SuperVOOC 100W, Android 13 OxygenOS, bateria 5000mAh, świetna wydajność w grach, cena 3599 zł",
    
    "Smartfon Motorola Edge 40 - procesor MediaTek Dimensity 8020, 8GB RAM, pamięć 256GB, ekran 6.55 cala pOLED 144Hz, aparat 50MP główny + 13MP ultra-wide, wodoodporność IP68, Android 13 czysty bez bloatware, ładowanie 68W, bateria 4400mAh, smukła obudowa 7.5mm, cena 2299 zł",
    
    "Smartfon Samsung Galaxy A54 5G - procesor Exynos 1380, 8GB RAM, pamięć 256GB, ekran 6.4 cala Super AMOLED 120Hz Full HD+, aparat 50MP główny + 12MP ultra-wide + 5MP macro, Android 13 One UI 5, bateria 5000mAh ładowanie 25W, IP67, świetny średniak z ekranem AMOLED, cena 1899 zł",
    
    "Smartfon realme GT Neo 5 - procesor Qualcomm Snapdragon 8+ Gen 1, 16GB RAM, pamięć 256GB, ekran 6.74 cala AMOLED 144Hz, aparat 50MP główny Sony IMX890, ultra-szybkie ładowanie 240W (pełne w 9.5 min!), Android 13 realme UI 4.0, bateria 4600mAh, najszybsze ładowanie na rynku, cena 2799 zł",
    
    # === SŁUCHAWKI (8) ===
    "Słuchawki Apple AirPods Pro 2 gen - prawdziwie bezprzewodowe TWS, aktywna redukcja szumów ANC z chipem H2, tryb Transparency, personalizowany Spatial Audio, etui USB-C z głośnikiem Find My, czas pracy 6h + 30h z etui, wodoodporność IPX4, automatyczne przełączanie między urządzeniami Apple, cena 1299 zł",
    
    "Słuchawki Sony WH-1000XM5 - nauszne bezprzewodowe, procesor V1 z najlepszą ANC na rynku, 8 mikrofonów redukcji szumów, LDAC i Hi-Res Audio, czas pracy 30h z ANC, szybkie ładowanie (3 min = 3h), automatyczna pauza przy zdejmowaniu, multipoint 2 urządzenia, miękkie poduszki, składana konstrukcja, cena 1699 zł",
    
    "Słuchawki Bose QuietComfort Earbuds II - TWS zCustomTune ANC dopasowaną do uszu, tryb Aware, doskonała jakość dźwięku Bose Signature, etui Bluetooth, czas pracy 6h + 18h z etui, wodoodporność IPX4, stabilne dopasowanie, aplikacja Bose Music, cena 1399 zł",
    
    "Słuchawki Jabra Elite 85t - TWS z regulowaną ANC (5 poziomów), 6 mikrofonów do rozmów, Hi-Fi dźwięk z regulowanym EQ, Qi wireless charging etui, czas pracy 5.5h + 25h z etui, multipoint 2 urządzenia, wodoodporność IPX4, kompaktowe etui, cena 799 zł",
    
    "Słuchawki SteelSeries Arctis Nova Pro Wireless - gamingowe bezprzewodowe z wymienną baterią (hot-swap!), aktywna redukcja szumów ANC, 360° Spatial Audio, ClearCast Gen 2 mikrofon, stacja ładująca z DAC, czas pracy 22h na baterii, obsługa Xbox/PS5/PC, regulowany RetractaBoom, premium jakość wykonania, cena 1899 zł",
    
    "Słuchawki HyperX Cloud III - przewodowe gamingowe, przetworniki 53mm, mikrofon DTS Headphone:X Spatial Audio, odłączany mikrofon z redukcją szumów, komfort dla długich sesji, aluminiowa konstrukcja, kabel USB-C i 3.5mm, kompatybilne PC/PS5/Xbox/Switch, świetna cena, cena 449 zł",
    
    "Słuchawki Sennheiser Momentum 4 Wireless - nauszne audiophile, Adaptive ANC, przetworniki 42mm Transducer, wsparcie aptX Adaptive, rekordowy czas pracy 60h!, szybkie ładowanie USB-C, aplikacja Smart Control z EQ, składana konstrukcja, etui w zestawie, doskonały dźwięk do muzyki, cena 1499 zł",
    
    "Słuchawki JBL Tune 230NC TWS - budżetowe z ANC, Ambient Aware i TalkThru, przetworniki 10mm Pure Bass, 4 mikrofony do rozmów, czas pracy 8h + 32h z etui, szybkie ładowanie USB-C, wodoodporność IPX4, aplikacja JBL Headphones, najlepszy stosunek ceny do możliwości z ANC, cena 399 zł",
    
    # === MYSZY (6) ===
    "Mysz Logitech MX Master 3S - bezprzewodowa ergonomiczna dla profesjonalistów, sensor 8000 DPI na każdej powierzchni, 7 przycisków programowalnych, MagSpeed scroll z trybem precyzyjnym, przycisk boczny, bateria 70 dni na ładowaniu, połączenie Bluetooth i USB receiver, Flow - sterowanie 3 komputerami, cicha praca 90% ciszej, cena 449 zł",
    
    "Mysz Razer DeathAdder V3 Pro - bezprzewodowa gamingowa dla FPS, sensor Focus Pro 30K DPI, przełączniki optyczne Gen-3 (90M kliknięć), HyperSpeed Wireless 4000Hz polling, waga 63g, bateria 90h, Razer Synapse 3, pokrycie gumowane, kształt ergonomiczny praworęczny, czas reakcji 0.25ms, dla profesjonalnych graczy esportowych, cena 699 zł",
    
    "Mysz SteelSeries Rival 3 - przewodowa gamingowa budżetowa, sensor TrueMove Core 8500 DPI, 6 przycisków programowalnych, podświetlenie RGB Prism z 3 strefami, przełączniki 60M kliknięć, kabel gumowy, chwyt uniwersalny, oprogramowanie SteelSeries Engine, świetna mysz początkowa do gier, lekka 77g, cena 149 zł",
    
    "Mysz Microsoft Surface Precision Mouse - bezprzewodowa biurowa premium, sensor 3200 DPI regulowany, 6 przycisków w tym kółko boczne, bateria 3 miesiące na jednej AA, Bluetooth i dongle USB, kompatybilna Windows/macOS, metalowa konstrukcja, chwyt ergonomiczny, doskonała precyzja do pracy z dokumentami, cena 399 zł",
    
    "Mysz Corsair Dark Core RGB Pro SE - bezprzewodowa gamingowa, sensor 18000 DPI, 8 przycisków programowalnych, ładowanie bezprzewodowe Qi, HyperSpeed Wireless, bateria 50h, podświetlenie RGB 9 stref, iCUE software, materiał Omron, chwyt dostosowany do ręki, waga 142g z opcją dociążenia, cena 549 zł",
    
    "Mysz Logitech G203 Lightsync - przewodowa gamingowa dla początkujących, sensor 8000 DPI, 6 przycisków programowalnych, podświetlenie RGB Lightsync z WAVE efektem, przełączniki mechaniczne, kabel 2.1m, G HUB software, symetryczny chwyt, lekka 85g, najlepsza budżetowa mysz gamingowa, cena 99 zł",
    
    # === KLAWIATURY (6) ===
    "Klawiatura Keychron K2 V2 - mechaniczna bezprzewodowa 75% layout, przełączniki Gateron Brown tactile, podświetlenie RGB white, połączenie Bluetooth 3 urządzenia + USB-C, bateria 240h, aluminiowa rama, hot-swappable switches, profil Mac/Windows, keycaps PBT doubleshot, kompaktowa dla biurka, cena 499 zł",
    
    "Klawiatura Corsair K70 RGB PRO - mechaniczna przewodowa gamingowa full-size, przełączniki Cherry MX Red liniowe, RGB per-key, podpórka pod nadgarstki w zestawie, kontrola multimediów, polling 8000Hz, iCUE software, aluminiowa rama, kabel USB-C odłączany, doskonała do gier i pisania, cena 799 zł",
    
    "Klawiatura Logitech MX Keys - bezprzewodowa membranowa dla profesjonalistów, podświetlenie smart auto-dimming, klawisze wklęsłe, Logitech Flow - 3 komputery, bateria 10 dni z podświetleniem / 5 miesięcy bez, USB-C ładowanie, klawiatura numeryczna, kompatybilna Windows/Mac/Linux, cicha praca, cena 549 zł",
    
    "Klawiatura HyperX Alloy Origins Core - mechaniczna przewodowa gamingowa TKL (bez numerycznej), przełączniki HyperX Red liniowe, RGB per-key Ngenuity software, aluminiowa rama wytrzymała, kabel USB-C odłączany, polling 1000Hz, kompaktowa 87 klawiszy, keycaps ABS pudding, świetna klawiatura esportowa, cena 449 zł",
    
    "Klawiatura Razer BlackWidow V3 Mini HyperSpeed - mechaniczna bezprzewodowa 65% layout, przełączniki Razer Green clicky, podświetlenie RGB Chroma, HyperSpeed Wireless + Bluetooth, bateria 200h, kabel USB-C odłączany, Razer Synapse 3, phantom keycaps, kompaktowa podróżna, wymienne switches, cena 699 zł",
    
    "Klawiatura SPC Gear GK630K Tournament - mechaniczna przewodowa budżetowa full-size, przełączniki Kailh Blue clicky, podświetlenie RGB 16.8M kolorów, polling 1000Hz, aluminiowa górna płyta, keycaps ABS doubleshot, oprogramowanie SPC Gear, kabel USB 1.8m, klawiatura numeryczna, najlepsza mechaniczna do 300 zł, cena 249 zł",
    
    # === MONITORY (5) ===
    "Monitor Dell UltraSharp U2723DE - 27 cali QHD 2560x1440 IPS, 100% sRGB i 95% DCI-P3, Delta E<2 kalibracja fabryczna, USB-C 90W Power Delivery, hub USB 4 porty, regulacja wysokości pivot swivel tilt, 60Hz, matryca 8-bit, doskonały do pracy z grafiką i projektowania, ramki 3-stronnie bezramkowe, cena 2199 zł",
    
    "Monitor LG 27GN950-B UltraGear - 27 cali 4K UHD 3840x2160 IPS Nano, 144Hz odświeżanie, G-Sync Compatible i FreeSync Premium Pro, HDR600, 98% DCI-P3, czas reakcji 1ms GtG, DisplayPort 1.4 + HDMI 2.1, Sphere Lighting 2.0 RGB, dla wymagających graczy 4K, pivot tilt, cena 3299 zł",
    
    "Monitor ASUS TUF Gaming VG27AQ - 27 cali WQHD 2560x1440 IPS, 165Hz overclockowane odświeżanie, G-Sync Compatible, czas reakcji 1ms MPRT, Extreme Low Motion Blur, GamePlus crosshair timer FPS counter, regulacja wysokości swivel tilt, 2x HDMI 2.0 + DisplayPort 1.2, głośniki stereo, świetny gaming w Full HD, cena 1499 zł",
    
    "Monitor Samsung Odyssey G7 C32G75T - 32 cale WQHD 2560x1440 VA zakrzywiony 1000R, 240Hz odświeżanie najszybsze, G-Sync Compatible i FreeSync Premium Pro, HDR600, czas reakcji 1ms GtG, QLED quantum dot 125% sRGB, CoreSync RGB podświetlenie tylne, dla hardcore graczy kompetytywnych, regulacja wysokości, cena 2799 zł",
    
    "Monitor BenQ SW270C PhotoVue - 27 cali QHD 2560x1440 IPS, 99% Adobe RGB i 95% DCI-P3, kalibracja sprzętowa, Delta E≤2, USB-C 60W, Hotkey Puck G2 kontroler, GamutDuo split screen, Paper Color Sync, 14-bit 3D LUT, matryca 10-bit, dla profesjonalnych fotografów i grafików, czytnik SD, cena 4499 zł",
    
    # === SMARTWATCHE (4) ===
    "Smartwatch Apple Watch Series 9 GPS 45mm - chip S9 SiP z 4-core Neural Engine, ekran Retina LTPO OLED Always-On, czujnik temperatury skóry, EKG + tętno + saturacja, Double Tap gesture, 50m wodoodporność, watchOS 10, bateria 18h, aluminiowa koperta, GPS precyzyjny, Siri on-device, kompatybilny iPhone, fitness tracking zaawansowany, cena 2199 zł",
    
    "Smartwatch Samsung Galaxy Watch6 Classic 47mm - procesor Exynos W930, ekran Super AMOLED 1.5 cala Always-On, obrotowa ramka fizyczna, czujniki BioActive (EKG, ciśnienie, body composition), GPS + GLONASS, 5ATM wodoodporność, Wear OS 4 powered by Samsung, bateria 425mAh 40h, kompatybilny Android, aluminiowa koperta, cena 1799 zł",
    
    "Smartwatch Garmin Fenix 7X Sapphire Solar - multisport outdoorowy premium, ekran 1.4 cala z ładowaniem solarnym, bateria 28 dni smartwatch / 122h GPS, 100+ trybów sportowych, mapy topograficzne, PulseOx tętno stamina, kompas barometr termometr, MIL-STD-810 wytrzymałość, szafirowe szkło, 100m wodoodporność, dla zaawansowanych sportowców, cena 3999 zł",
    
    "Smartwatch Amazfit GTS 4 - ekran AMOLED 1.75 cala HD Always-On, GPS + GLONASS, 150+ trybów sportowych, Zepp OS, bateria 8 dni, pomiar tętna SpO2 stres sen, Alexa wbudowana, 5ATM wodoodporność, 37g waga ultralekki, aluminiowa ramka, kompatybilny Android/iOS, świetny stosunek ceny do możliwości, cena 799 zł",
    
    # === TABLETY (3) ===
    "Tablet Apple iPad Air 5 gen - chip Apple M1 8-core, 8GB RAM, pamięć 256GB, ekran Liquid Retina 10.9 cala 2360x1640, aparat 12MP ultra-wide Center Stage, Touch ID w przycisku, USB-C, iPadOS 16, bateria 10h, kompatybilny Apple Pencil 2 + Magic Keyboard, dla kreatywnych profesjonalistów, cena 3499 zł",
    
    "Tablet Samsung Galaxy Tab S9 - procesor Qualcomm Snapdragon 8 Gen 2, 8GB RAM, pamięć 256GB, ekran Dynamic AMOLED 2X 11 cali 2560x1600 120Hz, aparat 13MP + 8MP ultra-wide, S Pen w zestawie, IP68 wodoodporność, 4 głośniki AKG Dolby Atmos, bateria 8400mAh 45W, DeX mode, Android 13 One UI 5, cena 3799 zł",
    
    "Tablet Lenovo Tab P11 Pro Gen 2 - procesor MediaTek Kompanio 1300T, 6GB RAM, pamięć 128GB, ekran OLED 11.5 cala 2560x1536, aparat 13MP + 8MP, 4 głośniki JBL, bateria 8000mAh, Android 12 czysty, Precision Pen 3 opcjonalnie, metalowa obudowa premium, świetny tablet multimedialny, cena 1999 zł",
    
    # === AKCESORIA (4) ===
    "Powerbank Anker PowerCore III Elite 25600mAh - pojemność 25600mAh 87Wh, ładowanie Power Delivery 60W USB-C dwukierunkowe, 2x USB-A 18W, wyświetlacz LED stanu baterii, ładuje laptopy MacBook Air, 3 urządzenia jednocześnie, zabezpieczenia MultiProtect, waga 568g, idealny w podróż, pass-through charging, cena 349 zł",
    
    "Ładowarka Ugreen Nexode 100W GaN - 3 porty (2x USB-C PD 100W + 1x USB-A QC 22.5W), technologia GaN kompaktowa, ładowanie 3 urządzeń jednocześnie, zabezpieczenia termiczne i przepięciowe, MacBook Pro 16 naładuje w 1.8h, składana wtyczka EU, ultraszybkie ładowanie telefonów i laptopów, cena 249 zł",
    
    "Kamera internetowa Logitech StreamCam - Full HD 1080p60 autofokus, korekta światła HD, AI framing Smart Auto-Framing, mikrofony stereo z redukcją szumów, montaż USB-C, tripod 1/4, Logitech Capture software, obraz pionowy i poziomy, dla streamerów i content creatorów, szkło optyczne premium, cena 799 zł",
    
    "Hub USB-C Anker PowerExpand 8-w-1 - 8 portów (HDMI 4K60, USB-C PD 85W, 2x USB-A 3.0, SD/microSD, Ethernet Gigabit), obsługa 2 monitorów, aluminiowa obudowa, pass-through charging laptopa, uniwersalny dla MacBook iPad Surface, transfer 5Gbps USB 3.0, kompaktowy design, wszystko w jednym hubie, cena 299 zł",
]

# %%
#declare connection with OpenAI
from openai import OpenAI
client = OpenAI(
    api_key=APIKey,
)

# %%
#function to generate embeddings 
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


# %%
#Create a database 
import pandas as pd
embeddings = [get_embedding(text, 'text-embedding-3-large') for text in knowledgebase]

df = pd.DataFrame({
    'Text': knowledgebase,
    'Vector': embeddings
})

# %%
df 

# %% [markdown]
# ## Queries

# %%
query = "laptop do gier RTX" 

# %%
query

# %%
query_vector = get_embedding(query, 'text-embedding-3-large')

# %%
from scipy.spatial.distance import cosine, euclidean
df['Cosine_Distance'] = df['Vector'].apply(lambda x: cosine(x, query_vector))

# %%
df.sort_values(by='Cosine_Distance')

# %%
selected_rows = df.nsmallest(2, 'Cosine_Distance')
print('Rezultat to:')
for reg in selected_rows['Text']:
    print(reg)
    
len(reg)

# %%
# Euclidean Distance
df['Euclidean_Distance'] = df['Vector'].apply(lambda x: euclidean(x, query_vector))
df.sort_values(by='Euclidean_Distance')

# %%
selected_rows1 = df.nsmallest(2, 'Euclidean_Distance')
print ('Rezultat to:')
for reg_1 in selected_rows1['Text']:
    print(reg_1)
len(reg_1)

# %%
# Sprawdzam statystyki
print("Cosine Distance stats:")
print(df['Cosine_Distance'].describe())
print("Euclidean Distance stats:")
print(df['Euclidean_Distance'].describe())

# Porównuję top-5 dla wszystkich 3 metryk
print("\n=== TOP-5 COMPARISON ===\n")

print("\1. Cosine Distance:")
print(df.nsmallest(5, 'Cosine_Distance')[['Text', 'Cosine_Distance']])

print("\n2. Euclidean Distance:")
print(df.nsmallest(5, 'Euclidean_Distance')[['Text', 'Euclidean_Distance']])

# %%


# %%


# %%



