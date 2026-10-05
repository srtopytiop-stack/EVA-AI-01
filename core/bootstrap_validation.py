def validate_returns_with_moving_block_bootstrap(
    returns: Sequence[float],
    *,
    bootstrap_replications: int = 2_000,
    block_length: int,
    seed: int | None = 42,
    risk_free_per_period: float = 0.0,
) -> BootstrapValidationReport:
    """
    Estimate empirical uncertainty for return statistics using MBB.

    The observed statistics are computed once from the supplied chronological
    returns. Bootstrap samples are generated only from those same returns.
    No strategy re-optimization occurs inside the bootstrap loop.

    A Sharpe statistic is undefined when the return variance is zero.
    StatisticalValidationError is translated into BootstrapValidationError at
    this boundary so callers can distinguish bootstrap-domain failure from
    the lower-level statistical implementation.
    """
    values = _normalise_returns(returns)
    risk_free = _validate_configuration(
        observations=len(values),
        bootstrap_replications=bootstrap_replications,
        block_length=block_length,
        seed=seed,
        risk_free_per_period=risk_free_per_period,
    )

    try:
        observed_sharpe = periodic_sharpe_ratio(
            values,
            risk_free_per_period=risk_free,
        )
    except StatisticalValidationError as exc:
        raise BootstrapValidationError(
            "observed returns are not suitable for Sharpe bootstrap inference: "
            f"{exc}"
        ) from exc

    observed_total_return = _total_return(values)
    observed_max_drawdown = _max_drawdown(values)

    rng = Random(seed)
    bootstrap_sharpes: list[float] = []
    bootstrap_total_returns: list[float] = []
    bootstrap_drawdowns: list[float] = []

    for replication in range(bootstrap_replications):
        sample = moving_block_bootstrap_sample(
            values,
            block_length=block_length,
            rng=rng,
        )

        try:
            bootstrap_sharpe = periodic_sharpe_ratio(
                sample,
                risk_free_per_period=risk_free,
            )
        except StatisticalValidationError as exc:
            raise BootstrapValidationError(
                "bootstrap sample "
                f"{replication + 1} produced undefined Sharpe inference: "
                f"{exc}"
            ) from exc

        bootstrap_sharpes.append(bootstrap_sharpe)
        bootstrap_total_returns.append(_total_return(sample))
        bootstrap_drawdowns.append(_max_drawdown(sample))

    sorted_sharpes = sorted(bootstrap_sharpes)
    sorted_total_returns = sorted(bootstrap_total_returns)
    sorted_drawdowns = sorted(bootstrap_drawdowns)

    return BootstrapValidationReport(
        observations=len(values),
        bootstrap_replications=bootstrap_replications,
        block_length=block_length,
        seed=seed,
        risk_free_per_period=risk_free,
        observed_periodic_sharpe=observed_sharpe,
        bootstrap_mean_sharpe=fmean(bootstrap_sharpes),
        bootstrap_std_sharpe=_sample_std(bootstrap_sharpes),
        sharpe_ci_lower=_quantile(sorted_sharpes, 0.025),
        sharpe_ci_upper=_quantile(sorted_sharpes, 0.975),
        bootstrap_positive_sharpe_fraction=sum(
            value > 0.0 for value in bootstrap_sharpes
        )
        / bootstrap_replications,
        observed_total_return=observed_total_return,
        bootstrap_mean_total_return=fmean(bootstrap_total_returns),
        bootstrap_std_total_return=_sample_std(bootstrap_total_returns),
        total_return_ci_lower=_quantile(sorted_total_returns, 0.025),
        total_return_ci_upper=_quantile(sorted_total_returns, 0.975),
        observed_max_drawdown=observed_max_drawdown,
        bootstrap_mean_max_drawdown=fmean(bootstrap_drawdowns),
        bootstrap_std_max_drawdown=_sample_std(bootstrap_drawdowns),
        max_drawdown_ci_lower=_quantile(sorted_drawdowns, 0.025),
        max_drawdown_ci_upper=_quantile(sorted_drawdowns, 0.975),
    )
