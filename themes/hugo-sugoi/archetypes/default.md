---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
draft: true
---

**Insert Lead paragraph here.**

## New Cool Posts

{{ range first 100 ( where .Site.RegularPages "Type" "cool" ) }}
* {{ .Title }}
{{ end }}