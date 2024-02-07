import numpy as np
import timeit


def test_logpdf(model, u1, u2):
    pdf_grid = model.pdf(u1, u2)
    logpdf_grid = model.logpdf(u1, u2)

    diff = np.log(pdf_grid) - logpdf_grid

    return np.sum(np.abs(diff))



def test_logpdf_speed(model, u1, u2, *params):
    # Define wrapper functions for pdf and np.exp(logpdf) to include the parameters
    def pdf_func():
        return model._pdf(u1, u2, *params)

    def exp_logpdf_func():
        return np.exp(model._logpdf(u1, u2, *params))

    # Number of executions to average over
    number_of_executions = 1000

    # Time the pdf function
    pdf_time = timeit.timeit(pdf_func, number=number_of_executions)
    print(f"Average execution time for pdf: {pdf_time / number_of_executions:.7f} seconds")

    # Time the np.exp(logpdf) function
    exp_logpdf_time = timeit.timeit(exp_logpdf_func, number=number_of_executions)
    print(f"Average execution time for np.exp(logpdf): {exp_logpdf_time / number_of_executions:.7f} seconds")

    # Compare speeds
    if pdf_time < exp_logpdf_time:
        print("pdf is faster.")
    else:
        print("np.exp(logpdf) is faster.")

    return pdf_time, exp_logpdf_time
