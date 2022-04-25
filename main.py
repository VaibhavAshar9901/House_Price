import pandas as pd
import pickle, joblib, lightgbm
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request
app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("lightgbm.pkl", "rb"))
pipe2 = joblib.load(open("naive_bayes.pkl", "rb"))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    loc = request.form.get('location')
    sqft = float(request.form.get('total_sqft'))
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    balcony = float(request.form.get('balcony'))
    gar_sqft = float(request.form.get('garage_sqft'))
    dict = {"1st Block Jayanagar": 1, "1st Phase JP Nagar": 2, "2nd Phase Judicial Layout": 3, "2nd Stage Nagarbhavi": 4, "5th Block Hbr Layout": 5, "5th Phase JP Nagar": 6, "6th Phase JP Nagar": 7, "7th Phase JP Nagar": 8, "8th Phase JP Nagar": 9, "9th Phase JP Nagar": 10, "AECS Layout": 11, "Abbigere": 12,
            "Akshaya Nagar": 13, "Ambalipura": 14, "Ambedkar Nagar": 15, "Amruthahalli": 16, "Anandapura": 17, "Ananth Nagar": 18, "Anekal": 19, "Anjanapura": 20, "Ardendale": 21, "Arekere": 22, "Attibele": 23, "BEML Layout": 24, "BTM 2nd Stage": 25, "BTM Layout": 26, "Babusapalaya": 27, "Badavala Nagar": 28, "Balagere": 29,
            "Banashankari": 30, "Banashankari Stage II": 31, "Banashankari Stage III": 32, "Banashankari Stage V": 33, "Banashankari Stage VI": 34, "Banaswadi": 35,"Banjara Layout": 36, "Bannerghatta": 37, "Bannerghatta Road": 38, "Basavangudi": 39, "Basaveshwara Nagar": 40, "Battarahalli": 41, "Begur": 42, "Begur Road": 43,
            "Bellandur": 44, "Benson Town": 45, "Bharathi Nagar": 46, "Bhoganhalli": 47, "Billekahalli": 48, "Binny Pete": 49, "Bisuvanahalli": 50, "Bommanahalli": 51, "Bommasandra": 52, "Bommasandra Industrial Area": 53, "Bommenahalli": 54, "Brookefield": 55, "Budigere": 56, "CV Raman Nagar": 57, "Chamrajpet": 58, "Chandapura": 59, "Channasandra": 60, "Chikka Tirupathi": 61, "Chikkabanavar": 62, "Chikkalasandra": 63, "Choodasandra": 64, "Cooke Town": 65, "Cox Town": 66,
            "Cunningham Road": 67, "Dasanapura": 68, "Dasarahalli": 69, "Devanahalli": 70, "Devarachikkanahalli": 71, "Dodda Nekkundi": 72, "Doddaballapur": 73, "Doddakallasandra": 74, "Doddathoguru": 75, "Domlur": 76, "Dommasandra": 77, "EPIP Zone": 78, "Electronic City": 79, "Electronic City Phase II": 80, "Electronics City Phase 1": 81, "Frazer Town": 82,
            "GM Palaya": 83, "Garudachar Palya": 84, "Giri Nagar": 85, "Gollarapalya Hosahalli": 86, "Gottigere": 87, "Green Glen Layout": 88, "Gubbalala": 89, "Gunjur": 90, "HAL 2nd Stage": 91, "HBR Layout": 92, "HRBR Layout": 93, "HSR Layout": 94, "Haralur Road": 95, "Harlur": 96, "Hebbal": 97, "Hebbal Kempapura": 98, "Hegde Nagar": 99, "Hennur": 100, "Hennur Road": 101,
            "Hoodi": 102, "Horamavu Agara": 103, "Horamavu Banaswadi": 104, "Hormavu": 105, "Hosa Road": 106, "Hosakerehalli": 107, "Hoskote": 108, "Hosur Road": 109, "Hulimavu": 110, "ISRO Layout": 111, "ITPL": 112, "Iblur Village": 113, "Indira Nagar": 114, "JP Nagar": 115, "Jakkur": 116, "Jalahalli": 117, "Jalahalli East": 118, "Jigani": 119, "Judicial Layout": 120, "KR Puram": 121,
            "Kadubeesanahalli": 122, "Kadugodi": 123, "Kaggadasapura": 124, "Kaggalipura": 125, "Kaikondrahalli": 126, "Kalena Agrahara": 127, "Kalyan nagar": 128, "Kambipura": 129, "Kammanahalli": 130, "Kammasandra": 131, "Kanakapura": 132, "Kanakpura Road": 133, "Kannamangala": 134, "Karuna Nagar": 135, "Kasavanhalli": 136, "Kasturi Nagar": 137,
            "Kathriguppe": 138, "Kaval Byrasandra": 139, "Kenchenahalli": 140, "Kengeri": 141, "Kengeri Satellite Town": 142, "Kereguddadahalli": 143, "Kodichikkanahalli": 144, "Kodigehalli": 145, "Kodihalli": 146, "Kogilu": 147, "Konanakunte": 148, "Koramangala": 149, "Kothannur": 150, "Kothanur": 151, "Kudlu": 152, "Kudlu Gate": 153, "Kumaraswami Layout": 154,
            "Kundalahalli": 155, "LB Shastri Nagar": 156, "Laggere": 157, "Lakshminarayana Pura": 158, "Lingadheeranahalli": 159, "Magadi Road": 160, "Mahadevpura": 161, "Mahalakshmi Layout": 162, "Mallasandra": 163, "Malleshpalya": 164, "Malleshwaram": 165, "Marathahalli": 166, "Margondanahalli": 167, "Marsur": 168, "Mico Layout": 169, "Munnekollal": 170, "Murugeshpalya": 171,
            "Mysore Road": 172, "NGR Layout": 173, "Nagarbhavi": 174, "Nagasandra": 175, "Nagavara": 176, "Nagavarapalya": 177, "Narayanapura": 178, "Neeladri Nagar": 179, "Nehru Nagar": 180, "OMBR Layout": 181, "Old Airport Road": 182, "Old Madras Road": 183, "Padmanabhanagar": 184, "Pai Layout": 185, "Panathur": 186, "Parappana Agrahara": 187, "Pattandur Agrahara": 188,
            "Poorna Pragna Layout": 189, "Prithvi Layout": 190, "R.T. Nagar": 191, "Rachenahalli": 192, "Raja Rajeshwari Nagar": 193, "Rajaji Nagar": 194, "Ramagondanahalli": 195, "Ramamurthy Nagar": 196, "Rayasandra": 197, "Sahakara Nagar": 198, "Sanjay nagar": 199, "Sarakki Nagar": 200, "Sarjapur": 201, "Sarjapur  Road": 202, "Sarjapura - Attibele Road": 203, "Sector 2 HSR Layout": 204,
            "Seegehalli": 205, "Shampura": 206, "Shivaji Nagar": 207, "Singasandra": 208, "Somasundara Palya": 209, "Sompura": 210, "Sonnenahalli": 211, "Subramanyapura": 212, "Sultan Palaya": 213, "TC Palaya": 214, "Talaghattapura": 215, "Thanisandra": 216, "Thigalarapalya": 217, "Thubarahalli": 218, "Thyagaraja Nagar": 219, "Tindlu": 220, "Tumkur Road": 221, "Ulsoor": 222, "Uttarahalli": 223, "Varthur": 224,
            "Varthur Road": 225, "Vasanthapura": 226, "Vidyaranyapura": 227, "Vijayanagar": 228, "Vishveshwarya Layout": 229, "Vishwapriya Layout": 230, "Vittasandra": 231, "Whitefield": 232, "Yelachenahalli": 233, "Yelahanka": 234, "Yelahanka New Town": 235, "Yelenahalli": 236, "Yeshwanthpur": 237, "other": 238}
    val = float(dict[loc])
    input = pd.DataFrame([[sqft, bath, bhk, balcony, gar_sqft, val]],
        columns=['total_sqft', 'bath', 'bhk', 'balcony', 'garage_sqft', 'location'])
    input2 = pd.DataFrame([[sqft, bath, bhk, balcony, gar_sqft]],
        columns=['total_sqft', 'bath', 'bhk', 'balcony', 'garage_sqft'])
    prediction = pipe.predict(input)[0] * 1e5
    prediction2 = pipe2.predict(input2)[0]
    if prediction2 == 0:
        return str(prediction2)
    else:
        return str(np.round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True, port=5000)