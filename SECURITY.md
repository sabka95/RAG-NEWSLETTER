# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

Please report security vulnerabilities to: security@example.com

## Security Features

- **OAuth2/OIDC Authentication** with Azure AD
- **Role-Based Access Control (RBAC)** with fine-grained permissions
- **Data Encryption** in transit and at rest
- **Secure Secret Management** with Azure Key Vault
- **RGPD Compliance** with data minimization and retention policies
- **Input Validation** and sanitization
- **Adversarial Robustness Testing** with ART framework

## Security Best Practices

1. **Environment Variables**: Never commit sensitive data to version control
2. **API Keys**: Use Azure Key Vault for production deployments
3. **Network Security**: Use HTTPS in production environments
4. **Access Control**: Implement proper RBAC based on user groups
5. **Audit Logging**: Enable comprehensive logging for security events
6. **Regular Updates**: Keep dependencies updated for security patches

## Dependencies Security

We regularly audit our dependencies for known vulnerabilities:
- Poetry lock file ensures reproducible builds
- Automated security scanning in CI/CD pipeline
- Regular updates of ML/AI frameworks
