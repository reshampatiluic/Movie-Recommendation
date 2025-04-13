pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                sh '''
                #!/bin/bash
                set -e

                echo Creating virtualenv...
                python -m venv .venv

                echo Installing requirements...
                source .venv/bin/activate
                python -m pip install --upgrade pip
                python -m pip install -r requirements.txt
                '''
            }
        }

        stage('Format') {
            steps {
                sh '''
                #!/bin/bash
                set -e

                source .venv/bin/activate
                echo Running linter and checking code format...
                python -m black --check .
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                #!/bin/bash
                set -e

                source .venv/bin/activate
                echo Running tests...
                python -m pytest
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                #!/bin/bash
                set -e

                echo Building docker image...
                docker compose build

                echo Spinning up container...
                docker compose down --remove-orphans
                docker compose up -d

                echo Successfully deployed!
                '''
            }
        }
    }

    post {
        always {
            sh '''
            #!/bin/bash
            set -e

            source .venv/bin/activate
            echo "Running coverage report..."
            pytest --cov=app tests/
            '''
        }
    }
}
