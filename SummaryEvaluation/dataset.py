from datasets import load_dataset

def fetch_summarization_datasets():
    # 1. Multi-News
    print("Downloading Multi-News (via Parquet mirror)...")
    multi_news = load_dataset("Awesome075/multi_news_parquet")

    # 2. XSum
    print("\nDownloading XSum...")
    xsum = load_dataset("xsum")

    # 3. WikiHow
    print("\nDownloading WikiHow (via cleaned mirror)...")
    wikihow = load_dataset("gursi26/wikihow-cleaned")

    # 4. CNN/DailyMail
    print("\nDownloading CNN/DailyMail...")
    cnndm = load_dataset("cnn_dailymail", "3.0.0")

    # 5. WCEP-10
    print("\nDownloading WCEP-10...")
    wcep10 = load_dataset("ccdv/WCEP-10")

    print("\n✅ All open datasets successfully downloaded and loaded into memory!")
    
    return multi_news, xsum, wikihow, cnndm, wcep10

# Execute the function
if __name__ == "__main__":
    multi_news, xsum, wikihow, cnndm, wcep10 = fetch_summarization_datasets()
