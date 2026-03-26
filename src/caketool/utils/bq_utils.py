from google.cloud import bigquery

ONE_GIGABYTE = 1024**3
ONE_TERABYTE = 1024 * ONE_GIGABYTE


def safety_query(
    query_statement: str, client: bigquery.Client | None = None, gb_limit=50, price_for_one_terabyte=8.44
) -> bigquery.QueryJob:
    """Execute a BigQuery SQL statement after validating its estimated scan size and cost.

    Performs a dry-run first to determine how many gigabytes the query will
    process.  If the estimated scan exceeds *gb_limit*, a ``ValueError`` is
    raised and the query is **not** executed.  Otherwise the query is run
    normally and the resulting ``QueryJob`` is returned.

    The estimated dollar cost (based on BigQuery on-demand pricing) is
    printed to stdout before execution.

    Parameters
    ----------
    query_statement : str
        Valid BigQuery Standard SQL query string.
    client : bigquery.Client, optional
        Authenticated BigQuery client instance.  Defaults to a module-level
        client created at import time.
    gb_limit : float, optional
        Maximum allowed data scan in gigabytes.  Queries that would process
        more than this amount are rejected.  Defaults to ``50``.
    price_for_one_terabyte : float, optional
        On-demand price per terabyte in USD, used only for the cost estimate
        printed to stdout.  Defaults to ``8.44`` (BigQuery on-demand pricing
        as of 2024).

    Returns
    -------
    bigquery.QueryJob
        The completed query job.  Call ``.to_dataframe()`` on the result to
        materialise the data locally.

    Raises
    ------
    ValueError
        If the estimated data scan exceeds *gb_limit*.

    Examples
    --------
    >>> job = safety_query("SELECT * FROM `project.dataset.table` LIMIT 1000")
    This query will process 0.002 Gigabytes.
    This query will process 0.0000168 dollars.
    >>> df = job.to_dataframe()
    """

    if client is None:
        client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=True)
    query_job = client.query(query_statement, job_config=job_config)
    bytes_processed = query_job.total_bytes_processed
    gigabytes_processed = bytes_processed / ONE_GIGABYTE
    print(f"This query will process {bytes_processed / ONE_GIGABYTE} Gigabytes.")
    print(f"This query will process {bytes_processed / ONE_TERABYTE * price_for_one_terabyte} dollars.")
    if gigabytes_processed >= gb_limit:
        raise ValueError(f"The data size exceeds the limitation >= {bytes_processed} GB")
    query_job = client.query(query_statement)
    return query_job
