DEFAULT_DATA = "data_motobikes.xlsx"
MODEL_PATH = "model_randomforest.pkl"

# Persist as Excel files (user requested Excel)
POSTS_SELL_XLSX = "posts_sell.xlsx"
POSTS_BUY_XLSX = "posts_buy.xlsx"
APPROVED_SELL_XLSX = "approved_posts_for_sale.xlsx"
APPROVED_BUY_XLSX = "approved_posts_for_buy.xlsx"
REJECTED_XLSX = "rejected_posts.xlsx"
QTV_ACCOUNTS = {
    "admin": "123456",
    "qtv1": "password1",
    "qtv2": "abc123"
}
# feature lists
num_cols = ['price_min', 'price_max', 'year_reg', 'km_driven', 'cc_numeric', 'price_segment_code', 'age']
flag_cols = ["is_moi", "is_do_xe", "is_su_dung_nhieu", "is_bao_duong", "is_do_ben", "is_phap_ly"]
cat_cols = ["brand", "vehicle_type", "model", "origin", "segment", 'engine_size']

BRANDS = ['Aprilia','Bmw','Bazan','Benelli','Brixton','Cr&S','Daelim','Detech','Ducati','Gpx','Halim',
          'Harley Davidson','Honda','Hyosung','Hãng Khác','Ktm','Kawasaki','Keeway','Kengo','Kymco',
          'Moto Guzzi','Nioshima','Peugeot','Piaggio','Rebelusa','Royal Enfield','Sym','Sachs','Sanda',
          'Suzuki','Taya','Triumph','Vento','Victory','Vinfast','Visitor','Yamaha']
