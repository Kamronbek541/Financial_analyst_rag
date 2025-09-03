import os
from sec_edgar_downloader import Downloader
from pathlib import Path

class SECDownloader:
    """
    A class to handle downloading of SEC filings.
    """
    def __init__(self, data_path: str, user_agent: str):

        self.user_agent = user_agent
        self.dl = Downloader(data_path, self.user_agent)
        print(f"SEC Downloader initialized. Intended save path: {Path(data_path).resolve()}")

    def download_latest_10k(self, ticker: str) -> str:
        """
        Downloads the most recent 10-K filing and deterministically finds the submission file
        based on the library's actual download path behavior.
        
        Args:
            ticker (str): The company ticker symbol (e.g., "AAPL").
            
        Returns:
            str: The path to the downloaded filing's submission file, or None if not found.
        """
        try:
            self.dl.get("10-K", ticker, limit=1)
            print(f"Successfully downloaded the latest 10-K for {ticker}.")

            base_path = Path("sec-edgar-filings") / ticker / "10-K"
            
            if not base_path.exists():
                raise FileNotFoundError(f"Base filing directory not found at '{base_path}'. Check library download behavior.")

            try:
                filing_dir = next(p for p in base_path.iterdir() if p.is_dir())
            except StopIteration:
                raise FileNotFoundError(f"No accession number directory found inside '{base_path}'.")

            submission_file_path = filing_dir / "full-submission.txt"
            
            if not submission_file_path.exists():
                raise FileNotFoundError(f"Submission file not found at '{submission_file_path}'")

            print(f"SUCCESS: Found primary submission file: {submission_file_path}")
            return str(submission_file_path)

        except Exception as e:
            print(f"An error occurred while downloading/finding the filing for {ticker}: {e}")
            return None

if __name__ == '__main__':
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    downloader = SECDownloader(
        data_path=config['data_path'], 
        user_agent=config['edgar_user_agent']
    )
    
    ticker_to_test = "MSFT"
    file_path = downloader.download_latest_10k(ticker_to_test)
    
    if file_path:
        print(f"\nSuccessfully located the primary document for {ticker_to_test} at: {file_path}")
    else:
        print(f"\nFailed to get the primary document for {ticker_to_test}.")