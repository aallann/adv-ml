import abc
import pandas as pd
import sqlalchemy as sql
import yfinance as yf
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker


class Database(abc.ABC):
    """Abstract base class databases synthesiser"""

    def __init__(
        self,
        host: str = "localhost",
        user: str = "root",
        password: str = " ",
        database: str = " ",
    ):
        self.__host = host
        self.__user = user
        self.__password = password
        self.__database = database

    @abc.abstractmethod
    def insert(self, df: pd.DataFrame, table_name: str):
        """Insert data into database"""
        return NotImplementedError

    @abc.abstractmethod
    def synth(self):
        """Build database"""
        return NotImplementedError


class OptionsDatabase(Database):
    """Financial database synthesiser to scrape & store options data from Yahoo Finance"""

    def __init__(
        self,
        host: str = "localhost",
        user: str = "root",
        password: str = " ",
        database: str = " ",
        params: dict = None,
        verbose: bool = False,
    ):

        if params is None:
            params = {
                "ticker": r"^SPX",
                "date": "2024-02-01",
                "period": "1d",
            }

            if verbose:
                print(f"Using default parameters: {params}")

        super().__init__(host, user, password, database)
        self.__ticker = params["ticker"]
        self.__date = params["date"]
        self.__period = params["period"]
        self.__engine = sql.create_engine(f"sqlite:///data/{database}.db")
        self.__metadata = sql.MetaData()
        self.__session = sessionmaker(bind=self.__engine)
        self.synth()

    def synth(self):
        """Build database inserting data"""
        tk = yf.Ticker(self.__ticker)
        exps = tk.options
        for e in exps:
            opt_chain = tk.option_chain(e)
            table_name_calls = f'{self.__ticker.replace("^", "")}_calls'
            table_name_puts = f'{self.__ticker.replace("^", "")}_puts'

            opt_chain.calls["expiration_date"] = e
            opt_chain.puts["expiration_date"] = e

            self.insert(opt_chain.calls, table_name_calls)
            self.insert(opt_chain.puts, table_name_puts)

        spot_price = tk.history(period="1d")["Close"].iloc[0]
        self.insert(
            pd.DataFrame(
                {
                    "date": [datetime.today()],
                    "ticker": [self.__ticker],
                    "spot_price": [spot_price],
                }
            ),
            "SPOT",
        )

    def insert(self, df: pd.DataFrame, table_name: str):
        """Insert data into database"""
        with self.scoper() as session:
            df.to_sql(table_name, self.__engine, if_exists="append", index=False)

    def read(self, table_name: str, condition: str, columns: list = None):
        """Read entries from database"""
        with self.scoper() as session:
            table = sql.Table(table_name, self.__metadata, autoload_with=self.__engine)
            if columns:
                query = sql.select(columns).where(sql.text(condition))
            else:
                query = sql.select(table).where(sql.text(condition))
            data = session.execute(query)
        return pd.DataFrame(data.fetchall(), columns=table.columns.keys())

    @contextmanager
    def scoper(self):
        """Limiting session scope for database operations.
        Context manager streamlines the process of creating and closing connection sessions.
        """
        session = self.__session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


class ModelDatabase(Database):
    """Stochastic model database object to store parameter data for synthetic data generation"""

    def __init__(
        self,
        host: str = "localhost",
        user: str = "root",
        password: str = " ",
        database: str = " ",
        model: str = "Heston",
        dataset: pd.DataFrame = None,
        verbose: bool = False,
    ):
        super().__init__(host, user, password, database)
        self.__model = model
        self.__table = f"{model}_params"
        self.__engine = sql.create_engine(f"sqlite:///data/{database}.db")
        self.__session = sessionmaker(bind=self.__engine)
        self.synth(dataset, database)

    def insert(self, df: pd.DataFrame, table_name: str):
        """Insert data into database"""
        df.to_sql(table_name, self.__engine, if_exists="append", index=False)

    def synth(self, dataset: pd.DataFrame, table_name: str):
        """Build database inserting data"""
        with self.scoper() as session:
            self.insert(dataset, table_name)

    def read(self, table_name: str, condition: str, columns: list = None):
        """Read entries from database"""
        with self.scoper() as session:
            table = sql.Table(table_name, self.__metadata, autoload_with=self.__engine)
            if columns:
                query = sql.select(columns).where(sql.text(condition))
            else:
                query = sql.select(table).where(sql.text(condition))
            data = session.execute(query)
        return pd.DataFrame(data.fetchall(), columns=table.columns.keys())

    @contextmanager
    def scoper(self):
        """Limiting session scope for database operations.
        Context manager streamlines the process of creating and closing connection sessions.
        """
        session = self.__session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
