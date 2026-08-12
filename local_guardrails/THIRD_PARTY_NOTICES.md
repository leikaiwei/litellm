# Third-party notices

## houbb/sensitive-word-data

- Source: <https://github.com/houbb/sensitive-word-data>
- Reviewed revision: `fe6fc2921836217b8c90619db81b24af8b22d80f`
- License: Apache License 2.0
- Local use: a manually curated and modified candidate subset for the Chinese abusive-language policy

The upstream dataset is not copied wholesale. The local full-message keyword list contains 21 exact-match
upstream candidates; three more reviewed upstream phrases appear only in anchored contextual patterns.
Other abbreviations, variants, and contexts are local additions. Upstream tags were not used as an abuse
taxonomy. The local matcher and its performance claims are independent of the upstream DFA implementation.

The Apache License 2.0 text is included at
[`third_party/sensitive-word-data-LICENSE.txt`](third_party/sensitive-word-data-LICENSE.txt).
