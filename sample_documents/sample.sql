-- Sample SQL Script for RAG Testing

CREATE TABLE Employees (
    EmpID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Department VARCHAR(50)
);

INSERT INTO Employees (EmpID, FirstName, LastName, Department)
VALUES (101, 'John', 'Doe', 'Engineering');

INSERT INTO Employees (EmpID, FirstName, LastName, Department)
VALUES (102, 'Jane', 'Smith', 'Marketing');

-- Query to retrieve all employees
SELECT * FROM Employees;
