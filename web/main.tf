# ==============================================================================
# CLOUD-BEREITSTELLUNG DER SCORING-API AUF MICROSOFT AZURE (Terraform, optional)
# ==============================================================================
# Legt an:
#   1. Ressourcengruppe          - Klammer um alles, am Ende mit "terraform destroy" gelöscht
#   2. Container Registry (ACR)  - privater Speicher für Docker-Image der API
#   3. Container Apps-Umgebung   - verwaltete Laufzeit für Container (kein eigener Server)
#   4. Container App "aml-api"   - laufende API mit öffentlicher HTTPS-Adresse
#   5. Budget-Alarm              - E-Mail, sobald 80 % des Monatsbudgets verbraucht sind
#
# 
# -----------------------------------------------------------------------------
# Ablauf (aus Ordner web, angemeldet mit "az login"):
# -----------------------------------------------------------------------------
#   az provider register --namespace Microsoft.App --wait         # nur einmal je Abonnement
#   terraform init
#   terraform apply -target=azurerm_container_registry.aml -target=azurerm_consumption_budget_resource_group.aml
#   docker build --platform linux/amd64 -f Dockerfile \
#       -t $(terraform output -raw registry)/aml-api:latest ..        # ".." = Projektordner als Build-Kontext
#   terraform output -raw registry_passwort | docker login $(terraform output -raw registry) \
#       -u $(terraform output -raw registry_benutzer) --password-stdin
#   docker push $(terraform output -raw registry)/aml-api:latest
#   terraform apply                                               # dann API selbst
#   -> Monitoring-Seite: terraform output monitoring_url
#   -> Simulator lokal gegen Cloud:
#      .venv/bin/python -m production.simulator --url <api_url> --anzahl 2000 --rate 20   (aus Projektordner)
#   terraform destroy                                             # danach alles wieder löschen (Kosten!)
#
# Werte für Variablen in einer Datei terraform.tfvars (nicht ins Repository):
#   subscription_id = "..."
#   email_budget    = "..."


# ==============================================================================
# 1. ANBIETER UND VARIABLEN
# ==============================================================================

# Lädt Bausteine für Azure und für Zufallswerte
terraform {
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 4.0" }
    random  = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

# Verbindet Terraform mit eigenem Azure-Abonnement
provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

# Azure-Abonnement (steht in terraform.tfvars)
variable "subscription_id" {
  description = "Azure-Abonnement (z. B. Azure for Students)"
  type        = string
}

# E-Mail für Budget-Alarm (steht in terraform.tfvars)
variable "email_budget" {
  description = "E-Mail-Adresse für den Budget-Alarm"
  type        = string
}

# Rechenzentrum in Paris (EU). Azure for Students erlaubt nur bestimmte Regionen:
# norwayeast, switzerlandnorth, francecentral, swedencentral, italynorth
# (Abfrage: az policy assignment list --query "[].parameters.listOfAllowedLocations.value")
variable "ort" {
  description = "Azure-Region"
  type        = string
  default     = "francecentral"
}

# Monatsbudget in Euro
variable "budget_euro" {
  description = "Monatsbudget, ab 80 % kommt eine Warnung"
  type        = number
  default     = 20
}

# Zufällige Endung, weil Name der Registry weltweit eindeutig sein muss
resource "random_string" "endung" {
  length  = 6
  upper   = false
  special = false
}


# ==============================================================================
# 2. RESSOURCEN
# ==============================================================================

# Ressourcengruppe: Klammer um alles
resource "azurerm_resource_group" "aml" {
  name     = "rg-aml-stream"
  location = var.ort
}

# Container Registry: privater Speicher für Docker-Image
resource "azurerm_container_registry" "aml" {
  name                = "acramlstream${random_string.endung.result}"
  resource_group_name = azurerm_resource_group.aml.name
  location            = azurerm_resource_group.aml.location
  sku                 = "Basic"
  admin_enabled       = true # einfache Anmeldung mit Benutzer/Passwort (für ein MVP ausreichend)
}

# Umgebung, in der Container Apps laufen
resource "azurerm_container_app_environment" "aml" {
  name                = "cae-aml-stream"
  resource_group_name = azurerm_resource_group.aml.name
  location            = azurerm_resource_group.aml.location
}

# API selbst
resource "azurerm_container_app" "api" {
  name                         = "aml-api"
  resource_group_name          = azurerm_resource_group.aml.name
  container_app_environment_id = azurerm_container_app_environment.aml.id
  revision_mode                = "Single"

  # Speichert Registry-Passwort geschützt
  secret {
    name  = "registry-passwort"
    value = azurerm_container_registry.aml.admin_password
  }

  # Sagt App, wo sie Image abholt
  registry {
    server               = azurerm_container_registry.aml.login_server
    username             = azurerm_container_registry.aml.admin_username
    password_secret_name = "registry-passwort"
  }

  template {
    # Genau eine Instanz: SQLite hat nur einen Schreiber (max. 1), und kein
    # Kaltstart in Demo (min. 1). Datenbank liegt im Container –
    # neue Transaktionen gehen bei einem Neustart verloren.
    min_replicas = 1
    max_replicas = 1

    # Container mit Image aus Registry
    container {
      name   = "api"
      image  = "${azurerm_container_registry.aml.login_server}/aml-api:latest"
      cpu    = 2.0 # 2 vCPU / 4 GiB: genug Platz für 1,7 GB große Datenbank im Container
      memory = "4Gi"
    }
  }

  # Macht API öffentlich per HTTPS erreichbar (leitet auf Port 8000 weiter)
  ingress {
    external_enabled = true # öffentlich erreichbar, Azure stellt HTTPS bereit
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
}

# Budget-Alarm: E-Mail ab 80 % des Monatsbudgets (Kostenkontrolle)
resource "azurerm_consumption_budget_resource_group" "aml" {
  name              = "budget-aml-stream"
  resource_group_id = azurerm_resource_group.aml.id
  amount            = var.budget_euro
  time_grain        = "Monthly"

  time_period {
    start_date = "2026-10-01T00:00:00Z" # muss der Erste eines Monats sein
  }

  notification {
    enabled        = true
    threshold      = 80
    operator       = "GreaterThan"
    contact_emails = [var.email_budget]
  }
}


# ==============================================================================
# 3. AUSGABEN NACH "terraform apply"
# ==============================================================================

# Adresse der Registry (für docker build / push)
output "registry" {
  value = azurerm_container_registry.aml.login_server
}

# Benutzername der Registry
output "registry_benutzer" {
  value = azurerm_container_registry.aml.admin_username
}

# Passwort der Registry (wird nur mit -raw angezeigt)
output "registry_passwort" {
  value     = azurerm_container_registry.aml.admin_password
  sensitive = true
}

# Öffentliche Adresse der API (für Simulator: --url)
output "api_url" {
  value = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

# Monitoring-Seite im Browser
output "monitoring_url" {
  value = "https://${azurerm_container_app.api.ingress[0].fqdn}/alarme"
}
