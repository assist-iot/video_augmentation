{{/*
Expand the name of the chart.
*/}}
{{- define "enabler.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "enabler.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "enabler.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Cilium Multi-cluster global service annotations.
*/}}
{{- define "globalServiceAnnotations" -}}
io.cilium/global-service: "true"
io.cilium/service-affinity: remote
{{- end }}

{{/*
Name of the component mediaserver.
*/}}
{{- define "mediaserver.name" -}}
{{- printf "%s-mediaserver" (include "enabler.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified component mediaserver name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "mediaserver.fullname" -}}
{{- printf "%s-mediaserver" (include "enabler.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}


{{/*
Component mediaserver labels.
*/}}
{{- define "mediaserver.labels" -}}
helm.sh/chart: {{ include "enabler.chart" . }}
{{ include "mediaserver.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Component mediaserver selector labels.
*/}}
{{- define "mediaserver.selectorLabels" -}}
app.kubernetes.io/name: {{ include "enabler.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
enabler: {{ .Chart.Name }}
app.kubernetes.io/component: mediaserver
isMainInterface: "yes"
tier: {{ .Values.mediaserver.tier }}
{{- end }}

{{/*
Name of the component runner.
*/}}
{{- define "runner.name" -}}
{{- printf "%s-runner" (include "enabler.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified component runner name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "runner.fullname" -}}
{{- printf "%s-runner" (include "enabler.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}


{{/*
Component runner labels.
*/}}
{{- define "runner.labels" -}}
helm.sh/chart: {{ include "enabler.chart" . }}
{{ include "runner.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Component runner selector labels.
*/}}
{{- define "runner.selectorLabels" -}}
app.kubernetes.io/name: {{ include "enabler.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
enabler: {{ .Chart.Name }}
app.kubernetes.io/component: runner
isMainInterface: "no"
tier: {{ .Values.runner.tier }}
{{- end }}

{{/*
Name of the component nginx.
*/}}
{{- define "nginx.name" -}}
{{- printf "%s-nginx" (include "enabler.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified component nginx name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "nginx.fullname" -}}
{{- printf "%s-nginx" (include "enabler.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}


{{/*
Component nginx labels.
*/}}
{{- define "nginx.labels" -}}
helm.sh/chart: {{ include "enabler.chart" . }}
{{ include "nginx.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Component nginx selector labels.
*/}}
{{- define "nginx.selectorLabels" -}}
app.kubernetes.io/name: {{ include "enabler.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
enabler: {{ .Chart.Name }}
app.kubernetes.io/component: nginx
isMainInterface: "no"
tier: {{ .Values.nginx.tier }}
{{- end }}

