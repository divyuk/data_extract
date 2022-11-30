# Notes data extract
The notes have a particular format - the headings and the URL to the Answer. The idea is to extract the Headings and the answer link encoded in the string text. The text is formed using the last part of the URL.
## Notes Format
The heading starts with # and then there is the link.

e.g.:

\#### Encoding techniques


https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/

## Result
1. Encoding techniques
    1. <a href="https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/
">types-of-categorical-data-encoding</a>

## Additional Details

The Notes also has some loose paragraphs which are combined at the end of the Result under the heading KEY POINTS.

e.g.

KEY POINTS
1. The OLS is a distance-minimizing approximation/estimation method, while MLE is a "likelihood" maximization method. Both are used to estimate the parameters of a linear regression model .OLS estimator needs no stochastic assumptions to provide its distance-minimizing solution but need assumptions for justifying that the estimator is BLUE, while MLE starts by assuming a joint probability density/mass function.